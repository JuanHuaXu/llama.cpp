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


def rows_to_h(batch, field, scale_field, device):
    q = torch.from_numpy(batch[field].astype(np.float32))
    scale = torch.from_numpy(batch[scale_field].astype(np.float32)).unsqueeze(1)
    return (q * scale).to(device=device, dtype=torch.float16)


def build_next_depth0_targets(records):
    targets = np.full(len(records), -1, dtype=np.int64)
    depth0_by_pos = {}
    for i, row in enumerate(records):
        if int(row["depth"]) != 0:
            continue
        key = (int(row["seq_id"]), int(row["pos"]))
        depth0_by_pos.setdefault(key, []).append(i)
    for vals in depth0_by_pos.values():
        vals.sort()

    for i, row in enumerate(records):
        if int(row["verified"]) != 1:
            continue
        vals = depth0_by_pos.get((int(row["seq_id"]), int(row["pos"]) + 1))
        if not vals:
            continue
        j = bisect.bisect_right(vals, i)
        if j < len(vals):
            targets[i] = vals[j]
    return targets


class StateLoRA(nn.Module):
    def __init__(self, n_embd, rank):
        super().__init__()
        self.down = nn.Linear(n_embd, rank, bias=False)
        self.up = nn.Linear(rank, n_embd, bias=False)
        self.scale = rank ** -0.5
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        return h.float() + self.up(self.down(h.float())) * self.scale


def sample_indices(acc_idx, rej_idx, batch_size, reject_frac, rng):
    if reject_frac <= 0.0 or len(rej_idx) == 0:
        return rng.choice(acc_idx, size=batch_size, replace=len(acc_idx) < batch_size)
    n_rej = min(batch_size, max(1, int(round(batch_size * reject_frac))))
    n_acc = batch_size - n_rej
    chosen_acc = rng.choice(acc_idx, size=n_acc, replace=len(acc_idx) < n_acc)
    chosen_rej = rng.choice(rej_idx, size=n_rej, replace=len(rej_idx) < n_rej)
    chosen = np.concatenate([chosen_acc, chosen_rej])
    rng.shuffle(chosen)
    return chosen


def sample_batch(records, targets, idx, device):
    batch = records[idx]
    target_batch = records[targets[idx]]
    h_out = rows_to_h(batch, "h_out_q", "h_out_scale", device)
    h_target = rows_to_h(target_batch, "h_in_q", "h_in_scale", device)
    return h_out, h_target


def loss_and_metrics(model, h_out, h_target, cos_weight, delta_norm_weight):
    pred = model(h_out)
    target = h_target.float()
    base = h_out.float()
    mse = F.mse_loss(pred, target)
    cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    delta_norm = (pred - base).float().norm(dim=-1).mean()
    loss = mse + cos_weight * cos_loss + delta_norm_weight * delta_norm
    with torch.no_grad():
        base_cos = F.cosine_similarity(base, target, dim=-1).mean()
        pred_cos = F.cosine_similarity(pred, target, dim=-1).mean()
        base_mse = F.mse_loss(base, target)
    return loss, mse, cos_loss, delta_norm, base_mse, base_cos, pred_cos


def evaluate(model, records, targets, idx, args, device, seed):
    if len(idx) == 0:
        return {}
    rng = np.random.default_rng(seed)
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            chosen = rng.choice(idx, size=args.batch_size, replace=len(idx) < args.batch_size)
            h_out, h_target = sample_batch(records, targets, chosen, device)
            vals.append([v.item() for v in loss_and_metrics(model, h_out, h_target, args.cos_weight, args.delta_norm_weight)])
    model.train()
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "loss": float(arr[:, 0].mean()),
        "mse": float(arr[:, 1].mean()),
        "cos_loss": float(arr[:, 2].mean()),
        "delta_norm": float(arr[:, 3].mean()),
        "base_mse": float(arr[:, 4].mean()),
        "base_cos": float(arr[:, 5].mean()),
        "pred_cos": float(arr[:, 6].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="Train an MTP hidden-state LoRA to map draft h_nextn toward verifier-aligned next h_in.")
    ap.add_argument("--state-dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reject-frac", type=float, default=0.0)
    ap.add_argument("--cos-weight", type=float, default=0.25)
    ap.add_argument("--delta-norm-weight", type=float, default=1e-4)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    header = read_state_header(args.state_dump)
    records = open_state(args.state_dump, header)
    targets = build_next_depth0_targets(records)
    mask = (targets >= 0) & (records["verified"] == 1)
    if args.depth >= 0:
        mask &= records["depth"] == args.depth
    accepted_mask = mask & (records["accepted"] == 1)
    rejected_mask = mask & (records["accepted"] == 0)
    idx = np.nonzero(mask)[0]
    acc_idx = np.nonzero(accepted_mask)[0]
    rej_idx = np.nonzero(rejected_mask)[0]
    if len(acc_idx) == 0:
        raise ValueError("no accepted rows selected")

    perm = rng.permutation(idx)
    n_eval = max(args.batch_size, len(perm) // 10)
    eval_idx = perm[:n_eval]
    eval_acc_idx = eval_idx[records[eval_idx]["accepted"] == 1]
    eval_rej_idx = eval_idx[records[eval_idx]["accepted"] == 0]
    train_idx = perm[n_eval:] if len(perm) > n_eval else perm
    train_acc_idx = train_idx[records[train_idx]["accepted"] == 1]
    train_rej_idx = train_idx[records[train_idx]["accepted"] == 0]
    if len(train_acc_idx) == 0:
        train_acc_idx = acc_idx

    model = StateLoRA(header["n_embd"], args.rank).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start_payload = {
        "event": "start",
        "args": vars(args),
        "records": int(len(records)),
        "selected": int(len(idx)),
        "selected_accepted": int(len(acc_idx)),
        "selected_rejected": int(len(rej_idx)),
        "train": int(len(train_idx)),
        "train_accepted": int(len(train_acc_idx)),
        "train_rejected": int(len(train_rej_idx)),
        "eval": int(len(eval_idx)),
        "eval_accepted": int(len(eval_acc_idx)),
        "eval_rejected": int(len(eval_rej_idx)),
    }
    print(json.dumps(start_payload), flush=True)
    print(json.dumps({
        "event": "eval_start",
        "all": evaluate(model, records, targets, eval_idx, args, device, args.seed + 1),
        "accepted": evaluate(model, records, targets, eval_acc_idx, args, device, args.seed + 2),
        "rejected": evaluate(model, records, targets, eval_rej_idx, args, device, args.seed + 3),
    }), flush=True)

    t0 = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for step in progress:
        chosen = sample_indices(train_acc_idx, train_rej_idx, args.batch_size, args.reject_frac, rng)
        h_out, h_target = sample_batch(records, targets, chosen, device)
        loss, mse, cos_loss, delta_norm, base_mse, base_cos, pred_cos = loss_and_metrics(
            model, h_out, h_target, args.cos_weight, args.delta_norm_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            progress.set_description(
                f"loss={loss.item():.4f} mse={mse.item():.4f} cos={pred_cos.item():.4f}/{base_cos.item():.4f} d={delta_norm.item():.2f}")

    eval_end = {
        "all": evaluate(model, records, targets, eval_idx, args, device, args.seed + 4),
        "accepted": evaluate(model, records, targets, eval_acc_idx, args, device, args.seed + 5),
        "rejected": evaluate(model, records, targets, eval_rej_idx, args, device, args.seed + 6),
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "format": "mtp-state-lora-v1",
        "args": vars(args),
        "state_header": header,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "eval_end": eval_end,
    }, args.out)
    print(json.dumps({"event": "done", "seconds": time.perf_counter() - t0, "eval_end": eval_end, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
