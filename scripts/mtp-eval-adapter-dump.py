#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_train_module():
    path = Path(__file__).with_name("mtp-train-fast-recursive-adapter.py")
    spec = importlib.util.spec_from_file_location("mtp_train_fast_recursive_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_adapter(value):
    if "=" in value:
        name, path = value.split("=", 1)
        return name, path
    path = value
    return Path(path).stem, path


def row_mask(records, token_to_local, depth):
    target = np.asarray(records["target_token"], dtype=np.int64)
    draft = np.asarray(records["draft_token"], dtype=np.int64)
    target_ok = (target >= 0) & (target < len(token_to_local)) & (token_to_local[target] >= 0)
    draft_ok = (draft >= 0) & (draft < len(token_to_local)) & (token_to_local[draft] >= 0)
    mask = (records["depth"] == depth) & (records["verified"] == 1) & target_ok & draft_ok
    if "row_type" in records.dtype.names:
        mask = mask & (records["row_type"] == 0)
    return mask


def summarize_rows(prefix, top, probs, target_local, draft_local, target_logits, draft_logits, p_min):
    if int(target_local.numel()) == 0:
        return {f"{prefix}_rows": 0}
    target_prob = probs.gather(1, target_local[:, None]).squeeze(1)
    draft_prob = probs.gather(1, draft_local[:, None]).squeeze(1)
    target_rank = (probs > target_prob[:, None]).sum(dim=1) + 1
    margin = target_logits - draft_logits
    return {
        f"{prefix}_rows": int(target_local.numel()),
        f"{prefix}_top1": float((top == target_local).float().mean().item()),
        f"{prefix}_target_p": float(target_prob.mean().item()),
        f"{prefix}_draft_p": float(draft_prob.mean().item()),
        f"{prefix}_continue": float((probs.max(dim=1).values >= p_min).float().mean().item()),
        f"{prefix}_target_rank_mean": float(target_rank.float().mean().item()),
        f"{prefix}_target_rank_le_5": float((target_rank <= 5).float().mean().item()),
        f"{prefix}_target_over_draft": float((target_logits > draft_logits).float().mean().item()),
        f"{prefix}_target_minus_draft_logit": float(margin.mean().item()),
    }


def add_weighted(out, counts, summary):
    for key, value in summary.items():
        if key.endswith("_rows"):
            counts[key] = counts.get(key, 0) + int(value)
            continue
        prefix = key.split("_", 1)[0]
        n = int(summary.get(f"{prefix}_rows", 0))
        out[key] = out.get(key, 0.0) + float(value) * n
        counts[key] = counts.get(key, 0) + n


def eval_model(train_mod, model, output_weight, records, idx, token_to_local, batch_size, device, p_min, base_top_by_pos=None):
    totals = {}
    counts = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            chosen = idx[start:start + batch_size]
            batch = records[chosen]
            h = train_mod.rows_to_h(batch, device)
            logits = train_mod.logits_for(model, output_weight, h).float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.argmax(probs, dim=-1)
            target_np = token_to_local[np.asarray(batch["target_token"], dtype=np.int64)]
            draft_np = token_to_local[np.asarray(batch["draft_token"], dtype=np.int64)]
            target_local = torch.from_numpy(target_np.astype(np.int64)).to(device=device)
            draft_local = torch.from_numpy(draft_np.astype(np.int64)).to(device=device)
            target_logits = logits.gather(1, target_local[:, None]).squeeze(1)
            draft_logits = logits.gather(1, draft_local[:, None]).squeeze(1)
            masks = {
                "all": np.ones(len(batch), dtype=bool),
                "accepted": np.asarray(batch["accepted"], dtype=np.int32) == 1,
                "rejected": np.asarray(batch["accepted"], dtype=np.int32) == 0,
            }
            for prefix, mask in masks.items():
                if not mask.any():
                    continue
                mask_t = torch.from_numpy(mask).to(device=device)
                add_weighted(
                    totals,
                    counts,
                    summarize_rows(
                        prefix,
                        top[mask_t],
                        probs[mask_t],
                        target_local[mask_t],
                        draft_local[mask_t],
                        target_logits[mask_t],
                        draft_logits[mask_t],
                        p_min,
                    ),
                )
            if base_top_by_pos is not None:
                base_top = torch.from_numpy(base_top_by_pos[start:start + len(batch)]).to(device=device)
                n = int(len(batch))
                totals["all_top_agree_base"] = totals.get("all_top_agree_base", 0.0) + float((top == base_top).float().mean().item()) * n
                counts["all_top_agree_base"] = counts.get("all_top_agree_base", 0) + n
    out = {}
    for key, n in counts.items():
        if key.endswith("_rows"):
            out[key] = int(n)
        else:
            out[key] = totals[key] / n if n else 0.0
    return out


def collect_top(train_mod, model, output_weight, records, idx, batch_size, device):
    tops = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = records[idx[start:start + batch_size]]
            h = train_mod.rows_to_h(batch, device)
            logits = train_mod.logits_for(model, output_weight, h).float()
            tops.append(torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64))
    return np.concatenate(tops) if tops else np.array([], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description="Evaluate MTP low-rank adapters on verifier accept dumps.")
    ap.add_argument("--accept-dump", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--output-cache", default="")
    ap.add_argument("--fr-vocab", default="")
    ap.add_argument("--runtime-depth", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--adapter", action="append", default=[], help="name=checkpoint.pt; repeatable. The implicit base model is always evaluated first.")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--input-normalized", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    train_mod = load_train_module()
    header = train_mod.read_accept_header(args.accept_dump)
    records = train_mod.open_accept(args.accept_dump, header)
    if "target_token" not in records.dtype.names:
        raise ValueError("adapter evaluation requires MTPACC3+ target_token rows")

    device = torch.device(args.device)
    reader = train_mod.GGUFReader(args.gguf)
    full_output_weight = train_mod.load_output_weight(reader, args.output_cache, device)
    fr_ids, token_to_local = train_mod.load_fr_vocab(args.fr_vocab, full_output_weight.shape[0])
    output_weight = full_output_weight[torch.from_numpy(fr_ids).to(device=device)]
    norm_weight = train_mod.load_norm_weight(reader, header["n_embd"], device)

    idx = np.nonzero(row_mask(records, token_to_local, args.runtime_depth - 1))[0]
    if args.max_rows > 0 and len(idx) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, size=args.max_rows, replace=False))

    print(json.dumps({
        "event": "start",
        "accept_dump": args.accept_dump,
        "runtime_depth": args.runtime_depth,
        "depth": args.runtime_depth - 1,
        "rows": int(len(idx)),
        "fr_vocab": int(len(fr_ids)),
        "records": int(header["records"]),
    }), flush=True)

    base = train_mod.LowRankHead(header["n_embd"], args.rank, norm_weight, args.input_normalized).to(device=device, dtype=torch.float32)
    base_top = collect_top(train_mod, base, output_weight, records, idx, args.batch_size, device)
    base_summary = eval_model(train_mod, base, output_weight, records, idx, token_to_local, args.batch_size, device, args.p_min)
    print(json.dumps({"event": "adapter_eval", "name": "base_grafted", **base_summary}), flush=True)

    for value in args.adapter:
        name, path = parse_adapter(value)
        model = train_mod.LowRankHead(header["n_embd"], args.rank, norm_weight, args.input_normalized).to(device=device, dtype=torch.float32)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        train_mod.load_adapter_into_model(model, ckpt["state_dict"])
        summary = eval_model(train_mod, model, output_weight, records, idx, token_to_local, args.batch_size, device, args.p_min, base_top)
        print(json.dumps({"event": "adapter_eval", "name": name, "path": path, **summary}), flush=True)


if __name__ == "__main__":
    main()
