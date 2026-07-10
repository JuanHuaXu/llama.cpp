#!/usr/bin/env python3
import argparse
import collections
import json
import os
import struct


HEADER = struct.Struct("<8sIIIIQ")
META = struct.Struct("<Qiiiiifiiiiff")


def main():
    ap = argparse.ArgumentParser(description="Inspect draft-MTP transition state dumps.")
    ap.add_argument("path")
    args = ap.parse_args()

    size = os.path.getsize(args.path)
    with open(args.path, "rb") as f:
        header = f.read(HEADER.size)
        magic, version, n_embd, meta, fmt, limit = HEADER.unpack(header)
        if magic != b"MTPST3\0\0" or version != 1 or meta != META.size or fmt != 1:
            raise SystemExit(f"unsupported state dump header: magic={magic!r} version={version} meta={meta} fmt={fmt}")
        record_size = meta + 2 * n_embd
        records = (size - HEADER.size) // record_size
        remainder = (size - HEADER.size) % record_size

        depth_counts = collections.defaultdict(collections.Counter)
        next_depth0 = collections.defaultdict(list)
        rows = []
        for i in range(records):
            rec = f.read(meta)
            batch, seq_id, pos, depth, prev_token, draft_token, prob, accepted, verified, n_acc, n_drafted, h_in_scale, h_out_scale = META.unpack(rec)
            f.seek(2 * n_embd, os.SEEK_CUR)
            row = {
                "i": i,
                "batch": batch,
                "seq_id": seq_id,
                "pos": pos,
                "depth": depth,
                "prev_token": prev_token,
                "draft_token": draft_token,
                "prob": prob,
                "accepted": accepted,
                "verified": verified,
                "n_acc": n_acc,
                "n_drafted": n_drafted,
                "h_in_scale": h_in_scale,
                "h_out_scale": h_out_scale,
            }
            rows.append(row)
            c = depth_counts[depth]
            c["rows"] += 1
            c["accepted"] += accepted
            c["verified_rejected"] += int((not accepted) and verified)
            c["tail"] += int(not verified)
            if depth == 0:
                next_depth0[(seq_id, pos)].append((i, prev_token))

    recover = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["verified"] != 1 or r["accepted"] != 0:
            continue
        d = r["depth"]
        recover[d]["verified_rejected"] += 1
        cands = [x for x in next_depth0.get((r["seq_id"], r["pos"] + 1), []) if x[0] > r["i"]]
        if cands:
            recover[d]["recovered_next_prev"] += 1

    print(json.dumps({
        "path": args.path,
        "size": size,
        "n_embd": n_embd,
        "record_size": record_size,
        "records": records,
        "remainder": remainder,
        "limit": limit,
    }, sort_keys=True))
    for depth in sorted(depth_counts):
        c = depth_counts[depth]
        denom = c["accepted"] + c["verified_rejected"]
        print(json.dumps({
            "depth": depth,
            "rows": c["rows"],
            "accepted": c["accepted"],
            "verified_rejected": c["verified_rejected"],
            "tail": c["tail"],
            "accept_all": c["accepted"] / c["rows"] if c["rows"] else 0.0,
            "accept_verified": c["accepted"] / denom if denom else 0.0,
            "reject_target_coverage": recover[depth]["recovered_next_prev"] / recover[depth]["verified_rejected"] if recover[depth]["verified_rejected"] else None,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
