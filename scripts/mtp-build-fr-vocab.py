#!/usr/bin/env python3
import argparse
import json
import os
import struct
from collections import Counter

import numpy as np


def read_mtpdump_header(path):
    with open(path, "rb") as f:
        header = f.read(36)
    magic, version, n_embd, n_labels, fmt, meta, limit = struct.unpack("<8sIIIIIQ", header)
    if magic != b"MTPDMP1\0" or version != 1 or fmt != 1 or n_labels != 3 or meta != 28:
        raise ValueError(f"unsupported MTP dump {path}: magic={magic!r} version={version} labels={n_labels} fmt={fmt} meta={meta}")
    rec_size = meta + n_embd
    records = (os.path.getsize(path) - 36) // rec_size
    return {"n_embd": n_embd, "records": records}


def open_mtpdump(path, header):
    dtype = np.dtype([
        ("seq_id", "<i4"),
        ("pos", "<i4"),
        ("token", "<i4"),
        ("label1", "<i4"),
        ("label2", "<i4"),
        ("label3", "<i4"),
        ("scale", "<f4"),
        ("q", "i1", (header["n_embd"],)),
    ])
    return np.memmap(path, mode="r", dtype=dtype, offset=36, shape=(header["records"],))


def read_accept_header(path):
    with open(path, "rb") as f:
        header = f.read(32)
    magic, version, n_embd, meta, fmt, limit = struct.unpack("<8sIIIIQ", header)
    if magic != b"MTPACC2\0" or version != 2 or fmt != 1 or meta != 52:
        raise ValueError(f"unsupported accept dump {path}: magic={magic!r} version={version} fmt={fmt} meta={meta}")
    records = (os.path.getsize(path) - 32) // (meta + n_embd)
    return {"n_embd": n_embd, "records": records}


def open_accept(path, header):
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


def add_counts(counter, values, chunk):
    for begin in range(0, len(values), chunk):
        part = np.asarray(values[begin:begin + chunk], dtype=np.int64)
        part = part[part >= 0]
        uniq, counts = np.unique(part, return_counts=True)
        counter.update(dict(zip(uniq.tolist(), counts.tolist())))


def main():
    ap = argparse.ArgumentParser(description="Build an FR-Spec token allowlist from saved MTP distillation dumps.")
    ap.add_argument("--dump", action="append", default=[], help="MTPDMP1 static/draft dump; may be repeated")
    ap.add_argument("--accept-dump", action="append", default=[], help="MTPACC2 accept/reject dump; may be repeated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--include-token", action="store_true", help="include current-token column from MTPDMP1 dumps")
    ap.add_argument("--label", action="append", choices=["label1", "label2", "label3"], default=[])
    ap.add_argument("--accepted-only", action="store_true", help="only count accepted draft tokens from accept dumps")
    ap.add_argument("--always-token", action="append", type=int, default=[])
    args = ap.parse_args()

    labels = args.label or ["label1", "label2", "label3"]
    counter = Counter()
    sources = []

    for path in args.dump:
        header = read_mtpdump_header(path)
        records = open_mtpdump(path, header)
        if args.include_token:
            add_counts(counter, records["token"], args.chunk)
        for label in labels:
            add_counts(counter, records[label], args.chunk)
        sources.append({"path": path, "records": int(header["records"]), "kind": "mtp_dump"})

    for path in args.accept_dump:
        header = read_accept_header(path)
        records = open_accept(path, header)
        if args.accepted_only:
            idx = np.nonzero((records["accepted"] == 1) & (records["verified"] == 1))[0]
            add_counts(counter, records["draft_token"][idx], args.chunk)
        else:
            add_counts(counter, records["draft_token"], args.chunk)
        sources.append({"path": path, "records": int(header["records"]), "kind": "accept_dump"})

    for tok in args.always_token:
        if tok >= 0:
            counter[tok] += 1 << 60

    top = counter.most_common(args.size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# token_id count rank\n")
        for rank, (tok, count) in enumerate(top):
            f.write(f"{int(tok)} # {int(count)} {rank}\n")

    print(json.dumps({"event": "done", "out": args.out, "size": len(top), "requested_size": args.size, "sources": sources}, indent=2), flush=True)


if __name__ == "__main__":
    main()
