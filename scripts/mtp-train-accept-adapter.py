#!/usr/bin/env python3
import argparse
import json
import math
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


def read_accept_dump_header(path):
    with open(path, "rb") as f:
        header = f.read(32)
    magic, version, n_embd, meta, fmt, limit = struct.unpack("<8sIIIIQ", header)
    if magic != b"MTPACC2\0":
        raise ValueError(f"bad accept dump magic: {magic!r}")
    if version != 2 or fmt != 1 or meta != 52:
        raise ValueError(f"unsupported accept dump header: version={version}, format={fmt}, meta={meta}")
    rec_size = meta + n_embd
    records = (os.path.getsize(path) - 32) // rec_size
    return {"n_embd": n_embd, "meta": meta, "record_size": rec_size, "records": records, "limit": limit}


def open_accept_dump(path, header):
    dtype = np.dtype([
        ("batch_id", "<u8"),
        ("seq_id", "<i4"),
        ("pos", "<i4"),
        ("depth", "<i4"),
        ("prev_token", "<i4"),
        ("draft_token", "<i4"),
        ("p", "<f4"),
        ("accepted", "<i4"),
        ("verified", "<i4"),
        ("n_accepted", "<i4"),
        ("n_drafted", "<i4"),
        ("scale", "<f4"),
        ("q", "i1", (header["n_embd"],)),
    ])
    return np.memmap(path, mode="r", dtype=dtype, offset=32, shape=(header["records"],))


def load_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    raise KeyError(name)


def load_output_weight(reader, cache_path, device):
    if cache_path and os.path.exists(cache_path):
        print(json.dumps({"event": "load_output_cache", "path": cache_path}), flush=True)
        weight = torch.from_numpy(np.asarray(np.load(cache_path, mmap_mode="r")))
    else:
        tensor = load_tensor(reader, "output.weight")
        print(json.dumps({"event": "dequant_output_weight", "shape": [int(d) for d in tensor.shape]}), flush=True)
        weight_np = dequantize(tensor.data, tensor.tensor_type).astype(np.float16)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, weight_np)
        weight = torch.from_numpy(weight_np)
    return weight.to(device=device, dtype=torch.float16, non_blocking=True)


def load_norm_weight(reader, n_embd, device):
    try:
        tensor = load_tensor(reader, "output_norm.weight")
        norm = torch.from_numpy(np.asarray(tensor.data, dtype=np.float32))
    except KeyError:
        norm = torch.ones(n_embd, dtype=torch.float32)
    return norm.to(device=device, dtype=torch.float16)


class LowRankHead(nn.Module):
    def __init__(self, n_embd, rank, norm_weight, input_normalized):
        super().__init__()
        self.input_normalized = input_normalized
        self.register_buffer("norm_weight", norm_weight.float().clone())
        self.down = nn.Linear(n_embd, rank, bias=False)
        self.up = nn.Linear(rank, n_embd, bias=False)
        self.scale = rank ** -0.5
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def base(self, h):
        h = h.float()
        if self.input_normalized:
            return h
        rms = torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + 1e-6)
        return h * rms * self.norm_weight

    def forward(self, h, return_parts=False):
        z_base = self.base(h)
        delta = self.up(self.down(z_base)) * self.scale
        z = z_base + delta
        if return_parts:
            return z, z_base, delta
        return z


def logits_for(model, output_weight, h, return_parts=False):
    if return_parts:
        z, z_base, delta = model(h, return_parts=True)
        logits = z.to(dtype=torch.float16) @ output_weight.T
        base_logits = z_base.to(dtype=torch.float16) @ output_weight.T
        return logits, base_logits, delta
    z = model(h)
    return z.to(dtype=torch.float16) @ output_weight.T


def build_indices(records, depth):
    depth_mask = records["depth"] == depth
    pos = np.nonzero(depth_mask & (records["verified"] == 1) & (records["accepted"] == 1))[0]
    neg = np.nonzero(depth_mask & (records["verified"] == 1) & (records["accepted"] == 0))[0]
    tail = np.nonzero(depth_mask & (records["verified"] == 0))[0]
    return pos, neg, tail


def sample_batch(records, pos_idx, neg_idx, batch_size, device, rng):
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("need both positives and verified negatives")
    n_pos = batch_size // 2
    n_neg = batch_size - n_pos
    idx = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=len(pos_idx) < n_pos),
        rng.choice(neg_idx, size=n_neg, replace=len(neg_idx) < n_neg),
    ])
    rng.shuffle(idx)
    batch = records[idx]
    q = torch.from_numpy(batch["q"].astype(np.float32))
    scale = torch.from_numpy(batch["scale"].astype(np.float32)).unsqueeze(1)
    h = (q * scale).to(device=device, dtype=torch.float16)
    tok = torch.from_numpy(batch["draft_token"].astype(np.int64)).to(device=device)
    target = torch.from_numpy(batch["accepted"].astype(np.float32)).to(device=device)
    return h, tok, target


def losses(logits, base_logits, delta, tok, target, p_min):
    logits_f = logits.float()
    logp = F.log_softmax(logits_f, dim=-1)
    draft_logp = logp.gather(1, tok[:, None]).squeeze(1)
    log_p_min = math.log(p_min)
    pos = target > 0.5
    neg = ~pos
    zero = logits_f.new_zeros(())
    pos_loss = -draft_logp[pos].mean() if pos.any() else zero
    neg_gate_loss = F.relu(draft_logp[neg] - log_p_min).mean() if neg.any() else zero
    base_logp = F.log_softmax(base_logits.float(), dim=-1)
    base_kl = F.kl_div(logp, base_logp.exp(), reduction="batchmean")
    delta_norm = delta.float().pow(2).mean()
    with torch.no_grad():
        probs = draft_logp.exp()
        pos_p = probs[pos].mean() if pos.any() else zero
        neg_p = probs[neg].mean() if neg.any() else zero
        neg_above = (probs[neg] >= p_min).float().mean() if neg.any() else zero
        pos_below = (probs[pos] < p_min).float().mean() if pos.any() else zero
    return pos_loss, neg_gate_loss, base_kl, delta_norm, pos_p, neg_p, neg_above, pos_below


def evaluate(model, output_weight, records, pos_idx, neg_idx, batch_size, batches, device, seed, p_min):
    model.eval()
    rng = np.random.default_rng(seed)
    sums = {"pos_loss":0.0,"neg_loss":0.0,"base_kl":0.0,"delta_norm":0.0,"pos_p":0.0,"neg_p":0.0,"neg_above_p_min":0.0,"pos_below_p_min":0.0}
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            h, tok, target = sample_batch(records, pos_idx, neg_idx, batch_size, device, rng)
            logits, base_logits, delta = logits_for(model, output_weight, h, return_parts=True)
            vals = losses(logits, base_logits, delta, tok, target, p_min)
            n = tok.numel(); total += n
            for key, val in zip(sums.keys(), vals):
                sums[key] += float(val.item()) * n
    model.train()
    return {key: value / total for key, value in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-dump", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--output-cache", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--depth", type=int, choices=[0,1,2], required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--positive-weight", type=float, default=0.20)
    ap.add_argument("--negative-weight", type=float, default=1.00)
    ap.add_argument("--base-kl-weight", type=float, default=0.03)
    ap.add_argument("--delta-norm-weight", type=float, default=0.0002)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--input-normalized", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    header = read_accept_dump_header(args.accept_dump)
    records = open_accept_dump(args.accept_dump, header)
    pos_idx, neg_idx, tail_idx = build_indices(records, args.depth)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(f"depth {args.depth} has pos={len(pos_idx)} neg={len(neg_idx)}")

    reader = GGUFReader(args.gguf)
    output_weight = load_output_weight(reader, args.output_cache, device)
    norm_weight = load_norm_weight(reader, header["n_embd"], device)
    model = LowRankHead(header["n_embd"], args.rank, norm_weight, args.input_normalized).to(device=device, dtype=torch.float32)
    init = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    model.load_state_dict(init["state_dict"], strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)

    print(json.dumps({"event":"start", "args": vars(args), "records": int(header["records"]), "n_embd": int(header["n_embd"]), "pos": int(len(pos_idx)), "neg": int(len(neg_idx)), "tail": int(len(tail_idx))}), flush=True)
    start_eval = evaluate(model, output_weight, records, pos_idx, neg_idx, args.batch_size, args.eval_batches, device, args.seed+1, args.p_min)
    print(json.dumps({"event":"eval_start", **start_eval}), flush=True)

    t0 = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for step in progress:
        h, tok, target = sample_batch(records, pos_idx, neg_idx, args.batch_size, device, rng)
        logits, base_logits, delta = logits_for(model, output_weight, h, return_parts=True)
        pos_loss, neg_loss, base_kl, delta_norm, pos_p, neg_p, neg_above, pos_below = losses(logits, base_logits, delta, tok, target, args.p_min)
        loss = args.positive_weight * pos_loss + args.negative_weight * neg_loss + args.base_kl_weight * base_kl + args.delta_norm_weight * delta_norm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            progress.set_description(f"loss={loss.item():.4f} pos_p={pos_p.item():.3f} neg_p={neg_p.item():.3f} neg_gate={neg_above.item():.3f} kl={base_kl.item():.3f}")

    end_eval = evaluate(model, output_weight, records, pos_idx, neg_idx, args.batch_size, args.eval_batches, device, args.seed+2, args.p_min)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"format":"mtp-direct-lowrank-v1", "args": vars(args), "dump_header": header, "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "eval_start": start_eval, "eval_end": end_eval}, args.out)
    print(json.dumps({"event":"done", "seconds": time.perf_counter()-t0, "eval_end": end_eval, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
