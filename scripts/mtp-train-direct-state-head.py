#!/usr/bin/env python3
import argparse
import bisect
import json
import os
import struct
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from gguf import GGUFReader
from gguf.quants import dequantize


HEADER = struct.Struct("<8sIIIIQ")


def read_state_header(path):
    with open(path, "rb") as f:
        magic, version, n_embd, meta, fmt, limit = HEADER.unpack(f.read(HEADER.size))
    if magic != b"MTPST3\0\0" or version != 1 or meta != 56 or fmt != 1:
        raise ValueError(f"unsupported state dump: magic={magic!r} version={version} meta={meta} fmt={fmt}")
    record_size = meta + 2 * n_embd
    records = (os.path.getsize(path) - HEADER.size) // record_size
    return {"n_embd": n_embd, "records": records, "record_size": record_size, "limit": limit}


def open_state(path, header):
    dtype = np.dtype([
        ("batch_id", "<u8"), ("seq_id", "<i4"), ("pos", "<i4"), ("depth", "<i4"),
        ("prev_token", "<i4"), ("draft_token", "<i4"), ("p", "<f4"),
        ("accepted", "<i4"), ("verified", "<i4"), ("n_accepted", "<i4"), ("n_drafted", "<i4"),
        ("h_in_scale", "<f4"), ("h_out_scale", "<f4"),
        ("h_in_q", "i1", (header["n_embd"],)),
        ("h_out_q", "i1", (header["n_embd"],)),
    ])
    return np.memmap(path, mode="r", dtype=dtype, offset=HEADER.size, shape=(header["records"],))


def load_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    raise KeyError(name)


def load_matrix(reader, name, cache_path, device):
    if cache_path and os.path.exists(cache_path):
        print(json.dumps({"event": "load_cache", "name": name, "path": cache_path}), flush=True)
        arr = np.load(cache_path, mmap_mode="r")
    else:
        tensor = load_tensor(reader, name)
        print(json.dumps({"event": "dequant_tensor", "name": name, "shape": [int(d) for d in tensor.shape]}), flush=True)
        arr = dequantize(tensor.data, tensor.tensor_type).astype(np.float16)
        if arr.shape[1] != 2048 and arr.shape[0] == 2048:
            arr = arr.T.copy()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, arr)
    return torch.from_numpy(np.asarray(arr)).to(device=device, dtype=torch.float16)


def load_vector(reader, name, n_embd, device):
    try:
        tensor = load_tensor(reader, name)
        arr = np.asarray(tensor.data, dtype=np.float32)
    except KeyError:
        arr = np.ones(n_embd, dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def rms_norm(x, weight):
    x = x.float()
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * weight


def rows_to_h(batch, field, scale_field, device):
    q = torch.from_numpy(batch[field].astype(np.float32))
    scale = torch.from_numpy(batch[scale_field].astype(np.float32)).unsqueeze(1)
    return (q * scale).to(device=device, dtype=torch.float16)


def build_recovered_targets(records, vocab_size):
    target = np.full(len(records), -1, dtype=np.int64)
    depth0_by_pos = {}
    for i, row in enumerate(records):
        if int(row["depth"]) != 0:
            continue
        key = (int(row["seq_id"]), int(row["pos"]))
        depth0_by_pos.setdefault(key, []).append((i, int(row["prev_token"])))
    for vals in depth0_by_pos.values():
        vals.sort()
    for i, row in enumerate(records):
        if int(row["verified"]) != 1 or int(row["accepted"]) != 0:
            continue
        vals = depth0_by_pos.get((int(row["seq_id"]), int(row["pos"]) + 1))
        if not vals:
            continue
        j = bisect.bisect_right(vals, (i, 2**31 - 1))
        if j >= len(vals):
            continue
        tok = vals[j][1]
        if 0 <= tok < vocab_size:
            target[i] = tok
    return target


class DirectStateHead(nn.Module):
    def __init__(self, n_embd, rank):
        super().__init__()
        self.down = nn.Linear(2 * n_embd, rank, bias=False)
        self.up = nn.Linear(rank, n_embd, bias=False)
        self.skip = nn.Linear(n_embd, n_embd, bias=False)
        self.scale = rank ** -0.5
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.eye_(self.skip.weight)

    def forward(self, concat, h_in_norm):
        return self.skip(h_in_norm.float()) + self.up(self.down(concat.float())) * self.scale


def sample_batch(records, idx, targets, token_embd, hnorm, enorm, batch_size, device, rng):
    chosen = rng.choice(idx, size=batch_size, replace=len(idx) < batch_size)
    batch = records[chosen]
    h_in = rows_to_h(batch, "h_in_q", "h_in_scale", device)
    h_out = rows_to_h(batch, "h_out_q", "h_out_scale", device)
    prev = torch.from_numpy(np.asarray(batch["prev_token"], dtype=np.int64)).to(device=device)
    tok = token_embd[prev]
    h_in_norm = rms_norm(h_in, hnorm)
    tok_norm = rms_norm(tok, enorm)
    concat = torch.cat([tok_norm, h_in_norm], dim=-1)
    y = torch.from_numpy(targets[chosen].astype(np.int64)).to(device=device)
    accepted = torch.from_numpy(np.asarray(batch["accepted"], dtype=np.float32)).to(device=device)
    return concat, h_in_norm, h_out, y, accepted


def sample_mixed_batch(records, acc_idx, rej_idx, any_idx, targets, token_embd, hnorm, enorm, batch_size, reject_frac, device, rng):
    if reject_frac <= 0 or len(rej_idx) == 0 or len(acc_idx) == 0:
        return sample_batch(records, any_idx, targets, token_embd, hnorm, enorm, batch_size, device, rng)
    n_rej = min(batch_size, max(1, int(round(batch_size * reject_frac))))
    n_acc = batch_size - n_rej
    chosen_rej = rng.choice(rej_idx, size=n_rej, replace=len(rej_idx) < n_rej)
    chosen_acc = rng.choice(acc_idx, size=n_acc, replace=len(acc_idx) < n_acc)
    chosen = np.concatenate([chosen_rej, chosen_acc])
    rng.shuffle(chosen)
    return sample_batch(records, chosen, targets, token_embd, hnorm, enorm, batch_size, device, rng)


def loss_and_metrics(model, output_weight, concat, h_in_norm, h_out, y, accepted, p_min, mse_weight):
    z = model(concat, h_in_norm)
    logits = z.to(dtype=output_weight.dtype) @ output_weight.T
    logits_f = logits.float()
    ce = F.cross_entropy(logits_f, y)
    if mse_weight > 0:
        accepted_mask = accepted > 0.5
        if accepted_mask.any():
            mse = F.mse_loss(z[accepted_mask].float(), h_out[accepted_mask].float())
        else:
            mse = torch.zeros((), device=logits.device)
    else:
        mse = torch.zeros((), device=logits.device)
    loss = ce + mse_weight * mse
    with torch.no_grad():
        probs = F.softmax(logits_f, dim=-1)
        top = probs.argmax(dim=-1)
        top1 = (top == y).float().mean()
        correct_p = probs.gather(1, y[:, None]).squeeze(1).mean()
        cont = (probs.max(dim=-1).values >= p_min).float().mean()
    return loss, ce, mse, top1, correct_p, cont


def evaluate(model, output_weight, records, idx, targets, token_embd, hnorm, enorm, args, device, seed):
    if len(idx) == 0:
        return {}
    rng = np.random.default_rng(seed)
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            concat, h_in_norm, h_out, y, accepted = sample_batch(records, idx, targets, token_embd, hnorm, enorm, args.batch_size, device, rng)
            loss, ce, mse, top1, correct_p, cont = loss_and_metrics(model, output_weight, concat, h_in_norm, h_out, y, accepted, args.p_min, args.mse_weight)
            vals.append((loss.item(), ce.item(), mse.item(), top1.item(), correct_p.item(), cont.item()))
    model.train()
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "loss": float(arr[:, 0].mean()),
        "ce": float(arr[:, 1].mean()),
        "mse": float(arr[:, 2].mean()),
        "top1": float(arr[:, 3].mean()),
        "correct_p": float(arr[:, 4].mean()),
        "continue": float(arr[:, 5].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="Train a lightweight direct MTP head from MTPST3 transition-state dumps.")
    ap.add_argument("--state-dump", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--output-cache", default="")
    ap.add_argument("--token-cache", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mse-weight", type=float, default=0.05)
    ap.add_argument("--reject-frac", type=float, default=0.0)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--depth", type=int, default=-1, help="-1 = all depths")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    header = read_state_header(args.state_dump)
    records = open_state(args.state_dump, header)
    reader = GGUFReader(args.gguf)
    output_weight = load_matrix(reader, "output.weight", args.output_cache, device)
    token_embd = load_matrix(reader, "token_embd.weight", args.token_cache, device)
    vocab_size = int(output_weight.shape[0])
    hnorm = load_vector(reader, "blk.40.nextn.hnorm.weight", header["n_embd"], device)
    enorm = load_vector(reader, "blk.40.nextn.enorm.weight", header["n_embd"], device)

    recovered = build_recovered_targets(records, vocab_size)
    targets = np.full(len(records), -1, dtype=np.int64)
    accepted = (records["verified"] == 1) & (records["accepted"] == 1)
    rejected = (records["verified"] == 1) & (records["accepted"] == 0) & (recovered >= 0)
    targets[accepted] = np.asarray(records["draft_token"][accepted], dtype=np.int64)
    targets[rejected] = recovered[rejected]
    mask = targets >= 0
    if args.depth >= 0:
        mask &= records["depth"] == args.depth
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        raise ValueError("no trainable rows selected")

    perm = rng.permutation(idx)
    n_eval = max(args.batch_size, len(perm) // 10)
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:] if len(perm) > n_eval else perm
    eval_acc_idx = eval_idx[records[eval_idx]["accepted"] == 1]
    eval_rej_idx = eval_idx[records[eval_idx]["accepted"] == 0]
    train_acc_idx = train_idx[records[train_idx]["accepted"] == 1]
    train_rej_idx = train_idx[records[train_idx]["accepted"] == 0]

    model = DirectStateHead(header["n_embd"], args.rank).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(json.dumps({
        "event": "start",
        "args": vars(args),
        "records": int(len(records)),
        "selected": int(len(idx)),
        "train": int(len(train_idx)),
        "eval": int(len(eval_idx)),
        "eval_accepted": int(len(eval_acc_idx)),
        "eval_rejected": int(len(eval_rej_idx)),
        "train_accepted": int(len(train_acc_idx)),
        "train_rejected": int(len(train_rej_idx)),
    }), flush=True)
    print(json.dumps({
        "event": "eval_start",
        "all": evaluate(model, output_weight, records, eval_idx, targets, token_embd, hnorm, enorm, args, device, args.seed + 1),
        "accepted": evaluate(model, output_weight, records, eval_acc_idx, targets, token_embd, hnorm, enorm, args, device, args.seed + 2),
        "rejected": evaluate(model, output_weight, records, eval_rej_idx, targets, token_embd, hnorm, enorm, args, device, args.seed + 3),
    }), flush=True)

    t0 = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for step in progress:
        concat, h_in_norm, h_out, y, acc = sample_mixed_batch(
            records, train_acc_idx, train_rej_idx, train_idx, targets,
            token_embd, hnorm, enorm, args.batch_size, args.reject_frac, device, rng)
        loss, ce, mse, top1, correct_p, cont = loss_and_metrics(model, output_weight, concat, h_in_norm, h_out, y, acc, args.p_min, args.mse_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            progress.set_description(
                f"loss={loss.item():.3f} ce={ce.item():.3f} mse={mse.item():.3f} top1={top1.item():.3f} cp={correct_p.item():.3f} cont={cont.item():.3f}")

    eval_end = {
        "all": evaluate(model, output_weight, records, eval_idx, targets, token_embd, hnorm, enorm, args, device, args.seed + 4),
        "accepted": evaluate(model, output_weight, records, eval_acc_idx, targets, token_embd, hnorm, enorm, args, device, args.seed + 5),
        "rejected": evaluate(model, output_weight, records, eval_rej_idx, targets, token_embd, hnorm, enorm, args, device, args.seed + 6),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "format": "mtp-direct-state-head-v1",
        "args": vars(args),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "eval_end": eval_end,
    }, args.out)
    print(json.dumps({"event": "done", "seconds": time.perf_counter() - t0, "eval_end": eval_end, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
