#!/usr/bin/env python3
import argparse
import json
import os
import struct

import numpy as np
import torch

from gguf import GGUFReader
from gguf.quants import dequantize


def load_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    raise KeyError(name)


def load_matrix(reader, name):
    tensor = load_tensor(reader, name)
    arr = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
    if arr.shape[1] != 2048 and arr.shape[0] == 2048:
        arr = arr.T.copy()
    return arr


def load_vector(reader, name, n_embd):
    try:
        tensor = load_tensor(reader, name)
        return np.asarray(tensor.data, dtype=np.float32)
    except KeyError:
        return np.ones(n_embd, dtype=np.float32)


def rms_norm_rows(x, weight):
    x = x.astype(np.float32, copy=False)
    scale = np.mean(x * x, axis=1, keepdims=True)
    return x * (1.0 / np.sqrt(scale + 1e-6)) * weight[None, :]


def main():
    ap = argparse.ArgumentParser(description="Export a trained DirectStateHead .pt to a llama.cpp MTP state-head binary.")
    ap.add_argument("--head", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-cache", default="")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--identity-skip", action="store_true", help="omit the learned dense skip matrix and use identity at runtime")
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    down = state["down.weight"].float().numpy()
    up = state["up.weight"].float().numpy()
    skip = state["skip.weight"].float().numpy()
    rank, two_n = down.shape
    n_embd = two_n // 2

    reader = GGUFReader(args.gguf)
    if args.token_cache and os.path.exists(args.token_cache):
        token_embd = np.asarray(np.load(args.token_cache, mmap_mode="r"), dtype=np.float32)
    else:
        token_embd = load_matrix(reader, "token_embd.weight")
    if token_embd.shape[1] != n_embd:
        raise ValueError(f"token embedding shape {token_embd.shape} does not match n_embd={n_embd}")

    enorm = load_vector(reader, "blk.40.nextn.enorm.weight", n_embd)
    hnorm = load_vector(reader, "blk.40.nextn.hnorm.weight", n_embd)
    down_e = np.ascontiguousarray(down[:, :n_embd])
    down_h = np.ascontiguousarray(down[:, n_embd:])

    n_vocab = token_embd.shape[0]
    token_down = np.empty((n_vocab, rank), dtype=np.float32)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    down_e_t = torch.from_numpy(down_e.T).to(device=device)
    enorm_t = torch.from_numpy(enorm).to(device=device)
    for beg in range(0, n_vocab, args.batch_size):
        end = min(n_vocab, beg + args.batch_size)
        batch = torch.from_numpy(np.asarray(token_embd[beg:end], dtype=np.float32)).to(device=device)
        e_norm = batch * torch.rsqrt(torch.mean(batch * batch, dim=1, keepdim=True) + 1e-6) * enorm_t
        token_down[beg:end] = (e_norm @ down_e_t).cpu().numpy()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"MTPDSH1\0")
        flags = 1 if args.identity_skip else 0
        f.write(struct.pack("<IIIIIf", 1, n_embd, rank, n_vocab, flags, float(rank ** -0.5)))
        np.asarray(hnorm, dtype="<f4").tofile(f)
        np.asarray(down_h, dtype="<f4").tofile(f)
        np.asarray(up, dtype="<f4").tofile(f)
        if not args.identity_skip:
            np.asarray(skip, dtype="<f4").tofile(f)
        np.asarray(token_down, dtype="<f4").tofile(f)

    print(json.dumps({
        "event": "exported",
        "head": args.head,
        "out": args.out,
        "n_embd": int(n_embd),
        "rank": int(rank),
        "n_vocab": int(n_vocab),
        "identity_skip": bool(args.identity_skip),
        "bytes": os.path.getsize(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()
