#!/usr/bin/env python3
import argparse
import json
import os
import random
import statistics
import struct
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Write a dense implementation plan for adding speculative decoding telemetry to a C++ inference server. Include edge cases and failure modes.",
    "Continue this Python module with robust tests and explanatory comments:\n\nclass ToolCallRouter:\n    def __init__(self):\n        self.handlers = {}\n",
    "Draft a long technical explanation of why multi-token prediction acceptance can collapse after the first token. Use equations where helpful.",
    "Write a fictional but technically precise incident report about a GPU inference service that regressed from 240 tok/s to 190 tok/s.",
    "Generate a JSON schema, examples, and validation notes for a tool-calling API used by local LLM agents.",
    "Create a detailed benchmark report comparing Q8 quantized inference, FP16 inference, MTP speculative decoding, and n-gram drafting.",
    "Write a Rust-style design document for a lock-free cache of token spans. Include invariants, tests, and performance risks.",
    "Continue the following story in a vivid analytical style, with lots of dialogue and internal reasoning:\n\nThe model stared at the profiler trace and realized the expensive part was not where anyone expected.",
    "Produce a long troubleshooting guide for llama.cpp CUDA crashes under aggressive GPU clocking and synchronization pressure.",
    "Write C++ pseudocode for a frequency-ranked vocab mask used during draft-token sampling. Include error handling and logging.",
    "Explain how to train a draft head by distillation using only a quantized teacher runtime. Include data formats and loss functions.",
    "Generate a mixed workload containing SQL queries, shell commands, JSON snippets, Markdown tables, and prose explanations.",
    "Write a detailed code review of a patch that adds prompt cache eviction by hit rate. Focus on subtle correctness bugs.",
    "Create a synthetic tool-use transcript where an assistant plans, runs commands, receives errors, fixes code, and validates performance.",
    "Explain the difference between accepted, verified rejected, and unverified tail tokens in speculative decoding data collection.",
    "Write a high-temperature brainstorm of ideas for accelerating dense transformer generation on dual RTX 5090 GPUs.",
    "Continue this TypeScript file with realistic types and tests:\n\nexport interface DraftTokenScore { token: number; probability: number; depth: number }\n",
    "Write a long-form guide to building a recursive MTP training dump format, including binary layout, append safety, and corruption checks.",
    "Generate 30 diverse prompts that would stress a speculative decoding system, then answer each one briefly.",
    "Write a mathematical derivation of expected speedup as a function of draft length, per-token acceptance, and draft overhead.",
]


ACCEPT_HEADER = struct.Struct("<8sIIIIQ")


def dump_records(path):
    if not path or not os.path.exists(path):
        return 0
    size = os.path.getsize(path)
    if size < ACCEPT_HEADER.size:
        return 0
    with open(path, "rb") as f:
        magic, version, n_embd, meta, fmt, _limit = ACCEPT_HEADER.unpack(f.read(ACCEPT_HEADER.size))
    if fmt != 1 or magic not in {b"MTPACC2\0", b"MTPACC3\0", b"MTPACC4\0", b"MTPACC5\0"}:
        return 0
    expected_meta = {2: 52, 3: 56, 4: 60, 5: 124}.get(version)
    if expected_meta != meta:
        return 0
    return (size - ACCEPT_HEADER.size) // (meta + n_embd)


def summarize_events(events):
    done = [e for e in events if e.get("event") == "request_done"]
    speeds = []
    accepted = 0
    drafted = 0
    tokens = 0
    for event in done:
        tok = int(event.get("tokens_predicted") or 0)
        tokens += tok
        predicted_ms = float(event.get("predicted_ms") or 0.0)
        if tok >= 700 and predicted_ms > 0:
            speeds.append(tok / (predicted_ms / 1000.0))
        accepted += int(event.get("draft_n_accepted") or 0)
        drafted += int(event.get("draft_n") or 0)
    return {
        "requests": len(done),
        "tokens_predicted": tokens,
        "median_tok_s": statistics.median(speeds) if speeds else None,
        "mean_tok_s": statistics.mean(speeds) if speeds else None,
        "draft_n": drafted,
        "draft_n_accepted": accepted,
        "accept_rate": accepted / drafted if drafted else None,
    }


def post_completion(url, prompt, n_predict, temperature, seed, timeout):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://10.254.0.1:11434")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--target-records", type=int, default=300000)
    ap.add_argument("--max-requests", type=int, default=240)
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    url = args.server.rstrip("/") + "/completion"
    started = dump_records(args.dump)
    events = []
    t0 = time.time()
    print(json.dumps({"event": "start", "dump": args.dump, "start_records": started, "target_records": args.target_records, "url": url}), flush=True)

    for i in range(args.max_requests):
        before = dump_records(args.dump)
        if before >= args.target_records:
            break
        base = PROMPTS[i % len(PROMPTS)]
        prompt = f"{base}\n\nRun id: {args.seed}-{i}. Be specific, continue naturally, and do not stop early."
        try:
            body = post_completion(url, prompt, args.n_predict, args.temperature, rng.randrange(1, 2**31), timeout=600)
            after = dump_records(args.dump)
            timings = body.get("timings") or {}
            event = {
                "event": "request_done",
                "i": i,
                "tokens_predicted": body.get("tokens_predicted"),
                "tokens_evaluated": body.get("tokens_evaluated"),
                "stop": body.get("stop"),
                "predicted_ms": timings.get("predicted_ms"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "prompt_ms": timings.get("prompt_ms"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
                "records_before": before,
                "records_after": after,
                "records_delta": after - before,
                "seconds": round(time.time() - t0, 3),
            }
            print(json.dumps(event), flush=True)
            events.append(event)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            event = {"event": "request_error", "i": i, "error": repr(exc), "records": dump_records(args.dump), "seconds": round(time.time() - t0, 3)}
            print(json.dumps(event), flush=True)
            events.append(event)
            time.sleep(5.0)
        time.sleep(args.sleep)

    final_records = dump_records(args.dump)
    summary = {
        "event": "done",
        "dump": args.dump,
        "start_records": started,
        "final_records": final_records,
        "records_added": final_records - started,
        "requests": len([e for e in events if e.get("event") == "request_done"]),
        "seconds": round(time.time() - t0, 3),
        **summarize_events(events),
    }
    print(json.dumps(summary), flush=True)
    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
