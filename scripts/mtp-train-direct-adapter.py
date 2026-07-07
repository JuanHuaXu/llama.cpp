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


def read_dump_header(path):
    with open(path, "rb") as f:
        header = f.read(36)
    magic, version, n_embd, n_labels, fmt, meta, limit = struct.unpack("<8sIIIIIQ", header)
    if magic != b"MTPDMP1\0":
        raise ValueError(f"bad dump magic: {magic!r}")
    if version != 1 or fmt != 1 or n_labels != 3 or meta != 28:
        raise ValueError(
            f"unsupported dump header: version={version}, format={fmt}, n_labels={n_labels}, meta={meta}"
        )
    rec_size = meta + n_embd
    n_records = (os.path.getsize(path) - 36) // rec_size
    return {
        "n_embd": n_embd,
        "n_labels": n_labels,
        "record_meta_bytes": meta,
        "record_size": rec_size,
        "records": n_records,
        "limit": limit,
    }


def open_dump(path, header):
    dtype = np.dtype(
        [
            ("seq_id", "<i4"),
            ("pos", "<i4"),
            ("token", "<i4"),
            ("label1", "<i4"),
            ("label2", "<i4"),
            ("label3", "<i4"),
            ("scale", "<f4"),
            ("q", "i1", (header["n_embd"],)),
        ]
    )
    return np.memmap(path, mode="r", dtype=dtype, offset=36, shape=(header["records"],))


def load_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    raise KeyError(name)


def load_output_weight(reader, cache_path, device):
    if cache_path and os.path.exists(cache_path):
        print(json.dumps({"event": "load_output_cache", "path": cache_path}), flush=True)
        weight = np.load(cache_path, mmap_mode="r")
        weight = torch.from_numpy(np.asarray(weight))
    else:
        tensor = load_tensor(reader, "output.weight")
        print(
            json.dumps(
                {
                    "event": "dequant_output_weight",
                    "shape": [int(dim) for dim in tensor.shape],
                    "tensor_type": int(tensor.tensor_type),
                    "n_bytes": int(tensor.n_bytes),
                }
            ),
            flush=True,
        )
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

    def forward(self, h):
        h = h.float()
        if self.input_normalized:
            z = h
        else:
            rms = torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + 1e-6)
            z = h * rms * self.norm_weight
        z = z + self.up(self.down(z)) * self.scale
        return z


def sample_batch(records, batch_size, label_name, device, rng):
    idx = rng.integers(0, len(records), size=batch_size)
    batch = records[idx]
    q = torch.from_numpy(batch["q"].astype(np.float32))
    scale = torch.from_numpy(batch["scale"].astype(np.float32)).unsqueeze(1)
    h = (q * scale).to(device=device, dtype=torch.float16)
    y = torch.from_numpy(batch[label_name].astype(np.int64)).to(device=device)
    return h, y


def evaluate(model, output_weight, records, label_name, batch_size, batches, device, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    with torch.no_grad():
        for _ in range(batches):
            h, y = sample_batch(records, batch_size, label_name, device, rng)
            z = model(h)
            logits = z.to(dtype=torch.float16) @ output_weight.T
            loss = F.cross_entropy(logits.float(), y)
            pred = torch.topk(logits, k=5, dim=-1).indices
            top1 += (pred[:, 0] == y).sum().item()
            top5 += (pred == y[:, None]).any(dim=-1).sum().item()
            total_loss += loss.item() * y.numel()
            total += y.numel()
    model.train()
    return {"loss": total_loss / total, "top1": top1 / total, "top5": top5 / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-cache", default="")
    parser.add_argument("--label", choices=["label1", "label2", "label3"], default="label1")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-normalized", action="store_true", help="treat dumped rows as already normalized for the shared output head")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    header = read_dump_header(args.dump)
    records = open_dump(args.dump, header)
    reader = GGUFReader(args.gguf)

    output_weight = load_output_weight(reader, args.output_cache, device)
    norm_weight = load_norm_weight(reader, header["n_embd"], device)
    model = LowRankHead(header["n_embd"], args.rank, norm_weight, args.input_normalized).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)

    print(
        json.dumps(
            {
                "event": "start",
                "dump": args.dump,
                "records": int(header["records"]),
                "n_embd": int(header["n_embd"]),
                "label": args.label,
                "rank": args.rank,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "device": str(device),
                "input_normalized": bool(args.input_normalized),
            }
        ),
        flush=True,
    )

    start_eval = evaluate(model, output_weight, records, args.label, args.batch_size, args.eval_batches, device, args.seed + 1)
    print(json.dumps({"event": "eval_start", **start_eval}), flush=True)

    t0 = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for step in progress:
        h, y = sample_batch(records, args.batch_size, args.label, device, rng)
        z = model(h)
        logits = z.to(dtype=torch.float16) @ output_weight.T
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            progress.set_description(f"loss={loss.item():.4f}")

    end_eval = evaluate(model, output_weight, records, args.label, args.batch_size, args.eval_batches, device, args.seed + 2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "format": "mtp-direct-lowrank-v1",
            "args": vars(args),
            "dump_header": header,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "eval_start": start_eval,
            "eval_end": end_eval,
        },
        args.out,
    )
    print(
        json.dumps(
            {
                "event": "done",
                "seconds": time.perf_counter() - t0,
                "eval_end": end_eval,
                "out": args.out,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
