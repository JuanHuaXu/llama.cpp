#!/usr/bin/env python3
import argparse
import bisect
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


def read_static_header(path):
    with open(path, "rb") as f:
        header = f.read(36)
    magic, version, n_embd, n_labels, fmt, meta, limit = struct.unpack("<8sIIIIIQ", header)
    if magic != b"MTPDMP1\0" or version != 1 or fmt != 1 or n_labels != 3 or meta != 28:
        raise ValueError(f"unsupported static dump: magic={magic!r} version={version} labels={n_labels} fmt={fmt} meta={meta}")
    records = (os.path.getsize(path) - 36) // (meta + n_embd)
    return {"n_embd": n_embd, "records": records}


def open_static(path, header):
    dtype = np.dtype([
        ("seq_id", "<i4"), ("pos", "<i4"), ("token", "<i4"),
        ("label1", "<i4"), ("label2", "<i4"), ("label3", "<i4"),
        ("scale", "<f4"), ("q", "i1", (header["n_embd"],)),
    ])
    return np.memmap(path, mode="r", dtype=dtype, offset=36, shape=(header["records"],))


def read_accept_header(path):
    with open(path, "rb") as f:
        header = f.read(32)
    magic, version, n_embd, meta, fmt, limit = struct.unpack("<8sIIIIQ", header)
    if magic != b"MTPACC2\0" or version != 2 or fmt != 1 or meta != 52:
        raise ValueError(f"unsupported accept dump: magic={magic!r} version={version} fmt={fmt} meta={meta}")
    records = (os.path.getsize(path) - 32) // (meta + n_embd)
    return {"n_embd": n_embd, "records": records}


def open_accept(path, header):
    dtype = np.dtype([
        ("batch_id", "<u8"), ("seq_id", "<i4"), ("pos", "<i4"), ("depth", "<i4"),
        ("prev_token", "<i4"), ("draft_token", "<i4"), ("p", "<f4"),
        ("accepted", "<i4"), ("verified", "<i4"), ("n_accepted", "<i4"), ("n_drafted", "<i4"),
        ("scale", "<f4"), ("q", "i1", (header["n_embd"],)),
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
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return weight.to(device=device, dtype=dtype, non_blocking=True)


def load_norm_weight(reader, n_embd, device):
    try:
        tensor = load_tensor(reader, "output_norm.weight")
        norm = torch.from_numpy(np.asarray(tensor.data, dtype=np.float32))
    except KeyError:
        norm = torch.ones(n_embd, dtype=torch.float32)
    return norm.to(device=device, dtype=torch.float16)


def load_fr_vocab(path, vocab_size):
    if not path:
        ids = np.arange(vocab_size, dtype=np.int64)
    else:
        vals = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                tok = int(line.split()[0])
                if 0 <= tok < vocab_size:
                    vals.append(tok)
        ids = np.array(sorted(set(vals)), dtype=np.int64)
    token_to_local = np.full(vocab_size, -1, dtype=np.int64)
    token_to_local[ids] = np.arange(len(ids), dtype=np.int64)
    return ids, token_to_local


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
        logits = z.to(dtype=output_weight.dtype) @ output_weight.T
        base_logits = z_base.to(dtype=output_weight.dtype) @ output_weight.T
        return logits, base_logits, delta
    z = model(h)
    return z.to(dtype=output_weight.dtype) @ output_weight.T


def rows_to_h(batch, device):
    q = torch.from_numpy(batch["q"].astype(np.float32))
    scale = torch.from_numpy(batch["scale"].astype(np.float32)).unsqueeze(1)
    return (q * scale).to(device=device, dtype=torch.float16)


def sample_static(records, idx, label, token_to_local, batch_size, device, rng):
    chosen = rng.choice(idx, size=batch_size, replace=len(idx) < batch_size)
    batch = records[chosen]
    h = rows_to_h(batch, device)
    y_np = token_to_local[np.asarray(batch[label], dtype=np.int64)]
    y = torch.from_numpy(y_np.astype(np.int64)).to(device=device)
    return h, y


def sample_accept(records, idx, token_to_local, batch_size, device, rng):
    chosen = rng.choice(idx, size=batch_size, replace=len(idx) < batch_size)
    batch = records[chosen]
    h = rows_to_h(batch, device)
    y_np = token_to_local[np.asarray(batch["draft_token"], dtype=np.int64)]
    y = torch.from_numpy(y_np.astype(np.int64)).to(device=device)
    return h, y


def sample_accept_targets(records, target_local, idx, batch_size, device, rng):
    chosen = rng.choice(idx, size=batch_size, replace=len(idx) < batch_size)
    batch = records[chosen]
    h = rows_to_h(batch, device)
    y = torch.from_numpy(target_local[chosen].astype(np.int64)).to(device=device)
    return h, y


def ce_loss(model, output_weight, h, y, p_min):
    logits, base_logits, delta = logits_for(model, output_weight, h, return_parts=True)
    logits_f = logits.float()
    ce = F.cross_entropy(logits_f, y)
    with torch.no_grad():
        probs = F.softmax(logits_f, dim=-1)
        top = torch.argmax(probs, dim=-1)
        top1 = (top == y).float().mean()
        correct_p = probs.gather(1, y[:, None]).squeeze(1).mean()
        continue_rate = (probs.max(dim=-1).values >= p_min).float().mean()
    base_logp = F.log_softmax(base_logits.float(), dim=-1)
    logp = F.log_softmax(logits_f, dim=-1)
    base_kl = F.kl_div(logp, base_logp.exp(), reduction="batchmean")
    delta_norm = delta.float().pow(2).mean()
    return ce, base_kl, delta_norm, top1, correct_p, continue_rate


def neg_gate_loss(model, output_weight, h, y, p_min):
    logits, base_logits, delta = logits_for(model, output_weight, h, return_parts=True)
    logp = F.log_softmax(logits.float(), dim=-1)
    draft_logp = logp.gather(1, y[:, None]).squeeze(1)
    loss = F.relu(draft_logp - math.log(p_min)).mean()
    with torch.no_grad():
        prob = draft_logp.exp()
        above = (prob >= p_min).float().mean()
        mean_p = prob.mean()
    base_logp = F.log_softmax(base_logits.float(), dim=-1)
    base_kl = F.kl_div(logp, base_logp.exp(), reduction="batchmean")
    delta_norm = delta.float().pow(2).mean()
    return loss, base_kl, delta_norm, above, mean_p


def build_reject_correct_targets(accept_records, token_to_local):
    if accept_records is None:
        return np.array([], dtype=np.int64)

    target_local = np.full(len(accept_records), -1, dtype=np.int64)
    depth0_by_next_pos = {}
    for i, row in enumerate(accept_records):
        if int(row["depth"]) != 0:
            continue
        key = (int(row["seq_id"]), int(row["pos"]))
        depth0_by_next_pos.setdefault(key, []).append((i, int(row["prev_token"])))

    for row in depth0_by_next_pos.values():
        row.sort()

    for i, row in enumerate(accept_records):
        if int(row["verified"]) != 1 or int(row["accepted"]) != 0:
            continue
        candidates = depth0_by_next_pos.get((int(row["seq_id"]), int(row["pos"]) + 1))
        if not candidates:
            continue
        j = bisect.bisect_right(candidates, (i, 2**31 - 1))
        if j >= len(candidates):
            continue
        token = candidates[j][1]
        if 0 <= token < len(token_to_local):
            target_local[i] = token_to_local[token]
    return target_local


def build_indices(static_records, accept_records, label, runtime_depth, token_to_local):
    static_idx = np.array([], dtype=np.int64)
    pos_idx = np.array([], dtype=np.int64)
    neg_idx = np.array([], dtype=np.int64)
    corr_idx = np.array([], dtype=np.int64)
    corr_targets = build_reject_correct_targets(accept_records, token_to_local)
    if static_records is not None:
        labels = np.asarray(static_records[label], dtype=np.int64)
        static_idx = np.nonzero((labels >= 0) & (labels < len(token_to_local)) & (token_to_local[labels] >= 0))[0]
    if accept_records is not None:
        toks = np.asarray(accept_records["draft_token"], dtype=np.int64)
        in_vocab = (toks >= 0) & (toks < len(token_to_local)) & (token_to_local[toks] >= 0)
        depth_mask = accept_records["depth"] == (runtime_depth - 1)
        verified = accept_records["verified"] == 1
        pos_idx = np.nonzero(depth_mask & verified & (accept_records["accepted"] == 1) & in_vocab)[0]
        neg_idx = np.nonzero(depth_mask & verified & (accept_records["accepted"] == 0) & in_vocab)[0]
        corr_idx = np.nonzero(depth_mask & verified & (accept_records["accepted"] == 0) & (corr_targets >= 0))[0]
    return static_idx, pos_idx, neg_idx, corr_idx, corr_targets


def eval_split(model, output_weight, static_records, accept_records, static_idx, pos_idx, neg_idx, corr_idx, corr_targets, label, token_to_local, args, device):
    rng = np.random.default_rng(args.seed + 99)
    out = {}
    model.eval()
    with torch.no_grad():
        if len(static_idx):
            h, y = sample_static(static_records, static_idx, label, token_to_local, args.batch_size, device, rng)
            ce, _, _, top1, correct_p, cont = ce_loss(model, output_weight, h, y, args.p_min)
            out.update(static_ce=float(ce.item()), static_top1=float(top1.item()), static_correct_p=float(correct_p.item()), static_continue=float(cont.item()))
        if len(pos_idx):
            h, y = sample_accept(accept_records, pos_idx, token_to_local, args.batch_size, device, rng)
            ce, _, _, top1, correct_p, cont = ce_loss(model, output_weight, h, y, args.p_min)
            out.update(rec_ce=float(ce.item()), rec_top1=float(top1.item()), rec_correct_p=float(correct_p.item()), rec_continue=float(cont.item()))
        if len(neg_idx):
            h, y = sample_accept(accept_records, neg_idx, token_to_local, args.batch_size, device, rng)
            loss, _, _, above, mean_p = neg_gate_loss(model, output_weight, h, y, args.p_min)
            out.update(neg_gate=float(loss.item()), neg_above=float(above.item()), neg_p=float(mean_p.item()))
        if len(corr_idx):
            h, y = sample_accept_targets(accept_records, corr_targets, corr_idx, args.batch_size, device, rng)
            ce, _, _, top1, correct_p, cont = ce_loss(model, output_weight, h, y, args.p_min)
            out.update(corr_ce=float(ce.item()), corr_top1=float(top1.item()), corr_correct_p=float(correct_p.item()), corr_continue=float(cont.item()))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description="Hybrid FastMTP-style adapter training from static and recursive MTP rows.")
    ap.add_argument("--static-dump", default="")
    ap.add_argument("--accept-dump", default="")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--output-cache", default="")
    ap.add_argument("--fr-vocab", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--runtime-depth", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--static-weight", type=float, default=0.35)
    ap.add_argument("--recursive-weight", type=float, default=1.0)
    ap.add_argument("--negative-weight", type=float, default=0.35)
    ap.add_argument("--reject-correct-weight", type=float, default=0.0)
    ap.add_argument("--base-kl-weight", type=float, default=0.02)
    ap.add_argument("--delta-norm-weight", type=float, default=0.0002)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--input-normalized", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not args.static_dump and not args.accept_dump:
        raise ValueError("provide --static-dump, --accept-dump, or both")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    label = f"label{args.runtime_depth}"

    static_records = None
    accept_records = None
    n_embd = None
    if args.static_dump:
        header = read_static_header(args.static_dump)
        static_records = open_static(args.static_dump, header)
        n_embd = header["n_embd"]
    if args.accept_dump:
        header = read_accept_header(args.accept_dump)
        accept_records = open_accept(args.accept_dump, header)
        if n_embd is not None and n_embd != header["n_embd"]:
            raise ValueError("static and accept dumps have different n_embd")
        n_embd = header["n_embd"]

    reader = GGUFReader(args.gguf)
    full_output_weight = load_output_weight(reader, args.output_cache, device)
    fr_ids, token_to_local = load_fr_vocab(args.fr_vocab, full_output_weight.shape[0])
    output_weight = full_output_weight[torch.from_numpy(fr_ids).to(device=device)]
    norm_weight = load_norm_weight(reader, n_embd, device)
    model = LowRankHead(n_embd, args.rank, norm_weight, args.input_normalized).to(device=device, dtype=torch.float32)
    if args.init_adapter:
        init = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        model.load_state_dict(init["state_dict"], strict=False)
        print(json.dumps({"event": "load_init_adapter", "path": args.init_adapter}), flush=True)
    else:
        print(json.dumps({"event": "zero_init_adapter"}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)

    static_idx, pos_idx, neg_idx, corr_idx, corr_targets = build_indices(static_records, accept_records, label, args.runtime_depth, token_to_local)
    if args.static_weight > 0 and len(static_idx) == 0:
        raise ValueError("static CE requested but no static rows survive the FR vocab")
    if args.recursive_weight > 0 and len(pos_idx) == 0:
        raise ValueError("recursive CE requested but no accepted recursive rows survive the FR vocab")
    if args.negative_weight > 0 and len(neg_idx) == 0:
        raise ValueError("negative gate requested but no verified rejected rows survive the FR vocab")
    if args.reject_correct_weight > 0 and len(corr_idx) == 0:
        raise ValueError("reject correction requested but no recoverable verified rejected rows survive the FR vocab")

    print(json.dumps({"event": "start", "args": vars(args), "n_embd": int(n_embd), "fr_vocab": int(len(fr_ids)), "static_rows": int(len(static_idx)), "recursive_pos": int(len(pos_idx)), "recursive_neg": int(len(neg_idx)), "recursive_corr": int(len(corr_idx))}), flush=True)
    print(json.dumps({"event": "eval_start", **eval_split(model, output_weight, static_records, accept_records, static_idx, pos_idx, neg_idx, corr_idx, corr_targets, label, token_to_local, args, device)}), flush=True)

    t0 = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for step in progress:
        loss = torch.zeros((), device=device)
        metrics = {}
        kl_terms = []
        dn_terms = []
        if args.static_weight > 0:
            h, y = sample_static(static_records, static_idx, label, token_to_local, args.batch_size, device, rng)
            ce, base_kl, delta_norm, top1, correct_p, cont = ce_loss(model, output_weight, h, y, args.p_min)
            loss = loss + args.static_weight * ce
            kl_terms.append(base_kl); dn_terms.append(delta_norm)
            metrics.update(sce=ce.item(), stop1=top1.item(), scp=correct_p.item(), scont=cont.item())
        if args.recursive_weight > 0:
            h, y = sample_accept(accept_records, pos_idx, token_to_local, args.batch_size, device, rng)
            ce, base_kl, delta_norm, top1, correct_p, cont = ce_loss(model, output_weight, h, y, args.p_min)
            loss = loss + args.recursive_weight * ce
            kl_terms.append(base_kl); dn_terms.append(delta_norm)
            metrics.update(rce=ce.item(), rtop1=top1.item(), rcp=correct_p.item(), rcont=cont.item())
        if args.negative_weight > 0:
            h, y = sample_accept(accept_records, neg_idx, token_to_local, args.batch_size, device, rng)
            gate, base_kl, delta_norm, above, mean_p = neg_gate_loss(model, output_weight, h, y, args.p_min)
            loss = loss + args.negative_weight * gate
            kl_terms.append(base_kl); dn_terms.append(delta_norm)
            metrics.update(ngate=gate.item(), nabove=above.item(), np=mean_p.item())
        if args.reject_correct_weight > 0:
            h, y = sample_accept_targets(accept_records, corr_targets, corr_idx, args.batch_size, device, rng)
            ce, base_kl, delta_norm, top1, correct_p, cont = ce_loss(model, output_weight, h, y, args.p_min)
            loss = loss + args.reject_correct_weight * ce
            kl_terms.append(base_kl); dn_terms.append(delta_norm)
            metrics.update(cce=ce.item(), ctop1=top1.item(), ccp=correct_p.item(), ccont=cont.item())
        if kl_terms:
            loss = loss + args.base_kl_weight * torch.stack(kl_terms).mean()
            loss = loss + args.delta_norm_weight * torch.stack(dn_terms).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            parts = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            progress.set_description(f"loss={loss.item():.4f} {parts}")

    eval_end = eval_split(model, output_weight, static_records, accept_records, static_idx, pos_idx, neg_idx, corr_idx, corr_targets, label, token_to_local, args, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"format": "mtp-direct-lowrank-v1", "args": vars(args), "fr_vocab_size": int(len(fr_ids)), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "eval_end": eval_end}, args.out)
    print(json.dumps({"event": "done", "seconds": time.perf_counter() - t0, "eval_end": eval_end, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
