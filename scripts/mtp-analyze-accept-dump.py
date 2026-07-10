#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import struct
from collections import Counter, defaultdict

import numpy as np


HEADER = struct.Struct("<8sIIIIQ")


def read_perf(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event") in {"sample", "request_done"}:
                rows.append(obj)
    full = [row for row in rows if int(row.get("tokens_predicted", 0)) >= 700]
    vals = []
    all_vals = []
    prev_seconds = 0.0
    for row in rows:
        tok = int(row.get("tokens_predicted", 0) or 0)
        tok_s = row.get("tok_s")
        if tok_s is None and row.get("predicted_ms"):
            ms = float(row["predicted_ms"])
            tok_s = tok / (ms / 1000.0) if ms > 0 else None
        if tok_s is None and row.get("seconds"):
            seconds = float(row["seconds"])
            dt = seconds - prev_seconds
            prev_seconds = seconds
            tok_s = tok / dt if dt > 0 else None
        if tok_s is None:
            continue
        all_vals.append(float(tok_s))
        if tok >= 700:
            vals.append(float(tok_s))
    draft_n = sum(int(row.get("draft_n") or 0) for row in rows)
    draft_n_accepted = sum(int(row.get("draft_n_accepted") or 0) for row in rows)
    return {
        "path": path,
        "samples": len(rows),
        "full_samples": len(full),
        "tokens_all": sum(int(row.get("tokens_predicted", 0)) for row in rows),
        "tokens_full": sum(int(row.get("tokens_predicted", 0)) for row in full),
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "accept_rate": draft_n_accepted / draft_n if draft_n else None,
        "median_all": statistics.median(all_vals) if all_vals else None,
        "median_full": statistics.median(vals) if vals else None,
        "mean_full": sum(vals) / len(vals) if vals else None,
        "min_full": min(vals) if vals else None,
        "max_full": max(vals) if vals else None,
    }


def open_accept_dump(path):
    with open(path, "rb") as f:
        magic, version, n_embd, meta, fmt, limit = HEADER.unpack(f.read(HEADER.size))
    if not (
        (magic == b"MTPACC2\0" and version == 2 and meta == 52)
        or (magic == b"MTPACC3\0" and version == 3 and meta == 56)
        or (magic == b"MTPACC4\0" and version == 4 and meta == 60)
        or (magic == b"MTPACC5\0" and version == 5 and meta == 124)
        or (magic == b"MTPACC6\0" and version == 6 and meta == 188)
    ) or fmt != 1:
        raise ValueError(f"unsupported accept dump {path}: magic={magic!r} version={version} meta={meta} fmt={fmt}")
    fields = [
        ("batch_id", "<u8"), ("seq_id", "<i4"), ("pos", "<i4"), ("depth", "<i4"),
        ("prev_token", "<i4"), ("draft_token", "<i4"), ("p", "<f4"),
        ("accepted", "<i4"), ("verified", "<i4"), ("n_accepted", "<i4"), ("n_drafted", "<i4"),
    ]
    if version >= 3:
        fields.append(("target_token", "<i4"))
    if version >= 4:
        fields.append(("row_type", "<i4"))
    if version >= 5:
        fields.extend([("candidate_ids", "<i4", (8,)), ("candidate_ps", "<f4", (8,))])
    if version >= 6:
        fields.extend([("target_candidate_ids", "<i4", (8,)), ("target_candidate_ps", "<f4", (8,))])
    fields.extend([("scale", "<f4"), ("q", "i1", (n_embd,))])
    dtype = np.dtype(fields)
    record_size = meta + n_embd
    records = (os.path.getsize(path) - HEADER.size) // record_size
    return np.memmap(path, mode="r", dtype=dtype, offset=HEADER.size, shape=(records,)), {
        "limit": int(limit),
        "version": int(version),
        "has_target_token": bool(version >= 3),
        "has_row_type": bool(version >= 4),
        "candidate_top_k": 8 if version >= 5 else 0,
        "target_candidate_top_k": 8 if version >= 6 else 0,
        "n_embd": int(n_embd),
        "record_size": int(record_size),
        "records": int(records),
    }


def analyze_dump(path):
    rows, header = open_accept_dump(path)
    if "row_type" in rows.dtype.names:
        is_draft = rows["row_type"] == 0
        is_stop = rows["row_type"] == 1
    else:
        is_draft = np.ones(len(rows), dtype=bool)
        is_stop = np.zeros(len(rows), dtype=bool)
    by_step = defaultdict(list)
    for i, row in enumerate(rows):
        by_step[int(row["batch_id"])].append(i)

    step_rows = []
    for idxs in by_step.values():
        step = rows[idxs]
        if "row_type" in step.dtype.names:
            step_is_draft = step["row_type"] == 0
            step_stops = int((step["row_type"] == 1).sum())
        else:
            step_is_draft = np.ones(len(step), dtype=bool)
            step_stops = 0
        draft = step[step_is_draft]
        verified = draft[draft["verified"] == 1]
        accepted = int(verified["accepted"].sum())
        step_rows.append((len(draft), accepted, int(step["n_accepted"].max()) if len(step) else 0, step_stops))

    steps = len(step_rows)
    accepted = sum(row[1] for row in step_rows)
    drafted = sum(row[0] for row in step_rows)
    stops = sum(row[3] for row in step_rows)
    generated = sum(row[1] + 1 for row in step_rows)
    out = {
        "path": path,
        **header,
        "steps": steps,
        "estimated_generated_tokens": generated,
        "accepted": accepted,
        "drafted": drafted,
        "confidence_stops": stops,
        "accepted_per_output": accepted / generated if generated else 0.0,
        "drafted_per_output": drafted / generated if generated else 0.0,
        "accepted_per_draft": accepted / drafted if drafted else 0.0,
        "output_per_step": generated / steps if steps else 0.0,
        "accepted_per_step": accepted / steps if steps else 0.0,
        "drafted_per_step": drafted / steps if steps else 0.0,
        "confidence_stops_per_step": stops / steps if steps else 0.0,
        "confidence_stops_per_output": stops / generated if generated else 0.0,
        "n_accepted_matches_rows": sum(1 for _, acc, max_acc, _ in step_rows if acc == max_acc) / steps if steps else 0.0,
        "n_accepted_hist": dict(sorted(Counter(row[1] for row in step_rows).items())),
        "n_drafted_hist": dict(sorted(Counter(row[0] for row in step_rows).items())),
        "confidence_stop_hist": dict(sorted(Counter(row[3] for row in step_rows).items())),
    }
    if "row_type" in rows.dtype.names:
        out["row_type_hist"] = {str(int(k)): int(v) for k, v in sorted(Counter(np.asarray(rows["row_type"], dtype=np.int32)).items())}
    if "candidate_ids" in rows.dtype.names and "target_token" in rows.dtype.names:
        verified_draft = is_draft & (rows["verified"] == 1) & (rows["target_token"] >= 0)
        if int(verified_draft.sum()):
            cand = np.asarray(rows["candidate_ids"][verified_draft], dtype=np.int64)
            target = np.asarray(rows["target_token"][verified_draft], dtype=np.int64)[:, None]
            draft = np.asarray(rows["draft_token"][verified_draft], dtype=np.int64)[:, None]
            target_hits = cand == target
            draft_hits = cand == draft
            target_any = target_hits.any(axis=1)
            draft_any = draft_hits.any(axis=1)
            target_rank = np.where(target_any, target_hits.argmax(axis=1) + 1, 0)
            draft_rank = np.where(draft_any, draft_hits.argmax(axis=1) + 1, 0)
            out.update({
                "candidate_target_in_topk": float(target_any.mean()),
                "candidate_draft_in_topk": float(draft_any.mean()),
                "candidate_target_rank_mean": float(target_rank[target_any].mean()) if target_any.any() else 0.0,
                "candidate_draft_rank_mean": float(draft_rank[draft_any].mean()) if draft_any.any() else 0.0,
            })
    if "target_candidate_ids" in rows.dtype.names and "target_token" in rows.dtype.names:
        verified_draft = is_draft & (rows["verified"] == 1) & (rows["target_token"] >= 0)
        if int(verified_draft.sum()):
            cand = np.asarray(rows["target_candidate_ids"][verified_draft], dtype=np.int64)
            target = np.asarray(rows["target_token"][verified_draft], dtype=np.int64)[:, None]
            draft = np.asarray(rows["draft_token"][verified_draft], dtype=np.int64)[:, None]
            target_hits = cand == target
            draft_hits = cand == draft
            target_any = target_hits.any(axis=1)
            draft_any = draft_hits.any(axis=1)
            target_rank = np.where(target_any, target_hits.argmax(axis=1) + 1, 0)
            draft_rank = np.where(draft_any, draft_hits.argmax(axis=1) + 1, 0)
            out.update({
                "target_candidate_target_in_topk": float(target_any.mean()),
                "target_candidate_draft_in_topk": float(draft_any.mean()),
                "target_candidate_target_rank_mean": float(target_rank[target_any].mean()) if target_any.any() else 0.0,
                "target_candidate_draft_rank_mean": float(draft_rank[draft_any].mean()) if draft_any.any() else 0.0,
            })
    for depth in [0, 1, 2]:
        mask = (rows["depth"] == depth) & is_draft
        stop_mask = (rows["depth"] == depth) & is_stop
        verified = mask & (rows["verified"] == 1)
        accepted_mask = verified & (rows["accepted"] == 1)
        row_count = int(mask.sum())
        verified_count = int(verified.sum())
        accepted_count = int(accepted_mask.sum())
        out[f"depth{depth}"] = {
            "rows": row_count,
            "verified": verified_count,
            "accepted": accepted_count,
            "tail": int((mask & (rows["verified"] == 0)).sum()),
            "accept_all": accepted_count / row_count if row_count else 0.0,
            "accept_verified": accepted_count / verified_count if verified_count else 0.0,
            "accepted_per_output": accepted_count / generated if generated else 0.0,
            "rows_per_output": row_count / generated if generated else 0.0,
            "mean_p": float(np.asarray(rows[mask]["p"], dtype=np.float32).mean()) if row_count else 0.0,
            "confidence_stops": int(stop_mask.sum()),
            "confidence_stop_mean_p": float(np.asarray(rows[stop_mask]["p"], dtype=np.float32).mean()) if int(stop_mask.sum()) else 0.0,
        }
        if "candidate_ids" in rows.dtype.names and "target_token" in rows.dtype.names and verified_count:
            cand = np.asarray(rows["candidate_ids"][verified], dtype=np.int64)
            target = np.asarray(rows["target_token"][verified], dtype=np.int64)[:, None]
            draft = np.asarray(rows["draft_token"][verified], dtype=np.int64)[:, None]
            out[f"depth{depth}"]["candidate_target_in_topk"] = float((cand == target).any(axis=1).mean())
            out[f"depth{depth}"]["candidate_draft_in_topk"] = float((cand == draft).any(axis=1).mean())
        if "target_candidate_ids" in rows.dtype.names and "target_token" in rows.dtype.names and verified_count:
            cand = np.asarray(rows["target_candidate_ids"][verified], dtype=np.int64)
            target = np.asarray(rows["target_token"][verified], dtype=np.int64)[:, None]
            draft = np.asarray(rows["draft_token"][verified], dtype=np.int64)[:, None]
            out[f"depth{depth}"]["target_candidate_target_in_topk"] = float((cand == target).any(axis=1).mean())
            out[f"depth{depth}"]["target_candidate_draft_in_topk"] = float((cand == draft).any(axis=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze llama.cpp MTPACC2/MTPACC3/MTPACC4/MTPACC5/MTPACC6 accept/reject dumps.")
    ap.add_argument("dump", nargs="+", help="MTPACC2/MTPACC3/MTPACC4/MTPACC5/MTPACC6 dump path(s)")
    ap.add_argument("--perf", action="append", default=[], help="optional perf JSONL to summarize; can be repeated")
    args = ap.parse_args()
    for path in args.dump:
        print(json.dumps({"event": "accept_dump", **analyze_dump(path)}, sort_keys=True))
    for path in args.perf:
        print(json.dumps({"event": "perf", **read_perf(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
