#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType, GGUFType, Keys


def main():
    ap = argparse.ArgumentParser(description="Export LowRankHead checkpoint as llama.cpp MTP hidden LoRA GGUF.")
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    down = np.ascontiguousarray(state["down.weight"].float().numpy().astype(np.float16))
    up = state["up.weight"].float().numpy()
    norm = state["norm_weight"].float().numpy()
    rank, n_embd = down.shape
    if up.shape != (n_embd, rank) or norm.shape != (n_embd,):
        raise ValueError(f"unexpected checkpoint shapes: down={down.shape} up={up.shape} norm={norm.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(out, "qwen35moe")
    writer.add_architecture()
    writer.add_type(GGUFType.ADAPTER)
    writer.add_string(Keys.Adapter.TYPE, "lora")
    # llama.cpp applies alpha/rank. sqrt(rank) matches LowRankHead's rank^-0.5 scale.
    writer.add_float32(Keys.Adapter.LORA_ALPHA, math.sqrt(rank))
    writer.add_tensor("mtp_hidden.lora_a", down, raw_dtype=GGMLQuantizationType.F16)
    # llama.cpp applies alpha/rank. This pre-scale makes the effective update rank^-1.
    up = np.ascontiguousarray((up * (rank ** -0.5)).astype(np.float16))
    writer.add_tensor("mtp_hidden.lora_b", up, raw_dtype=GGMLQuantizationType.F16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(json.dumps({"event": "exported", "head": args.head, "out": str(out), "rank": rank, "n_embd": n_embd, "alpha": math.sqrt(rank)}))


if __name__ == "__main__":
    main()
