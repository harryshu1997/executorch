"""
Gemma-4-E2B-it server-tier benchmark on A6000.

Measures:
  - prefill latency vs prompt length (1, 32, 128, 512, 1024 tokens)
  - decode rate (tokens/s) for 50-token outputs
  - peak VRAM
  - per-frame energy (server compute) via nvidia-smi power.draw sampled at 1 Hz

Outputs a JSON file for the cost table:
  cost[(LLM_query, server)] = (latency_ms, energy_J, output_bytes)
where output_bytes = ~250 KB for a 50-token response (BPE-encoded answer
+ small JSON envelope), and latency_ms / energy_J are the prefill+decode
totals at a representative prompt length.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"


@dataclass
class BenchRow:
    prompt_tokens: int
    decode_tokens: int
    prefill_ms: float
    decode_ms: float
    decode_tok_per_s: float
    peak_vram_mb: float
    avg_power_w: float
    energy_J: float


def gpu_power_sampler(stop_event: threading.Event,
                      samples: list[float],
                      gpu_index: int = 0) -> None:
    while not stop_event.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--id={gpu_index}",
                 "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            samples.append(float(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.5)


def run_one(model, tokenizer, device, prompt_tokens: int,
            decode_tokens: int = 50, gpu_index: int = 0) -> BenchRow:
    # Build a synthetic prompt of approx prompt_tokens length.
    base_prompt = "Describe the kitchen scene in concrete sensory detail. "
    text = (base_prompt * 200)[:prompt_tokens * 6]  # rough chars-per-token=6
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=prompt_tokens).to(device)
    actual_prompt_tokens = enc["input_ids"].shape[1]

    # Power sampler.
    samples: list[float] = []
    stop = threading.Event()
    t = threading.Thread(target=gpu_power_sampler, args=(stop, samples, gpu_index))
    t.start()

    torch.cuda.reset_peak_memory_stats(device)

    # Prefill
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out_pre = model(**enc, use_cache=True)
        past = out_pre.past_key_values
        next_id = out_pre.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    prefill_ms = 1000 * (time.time() - t0)

    # Decode
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(decode_tokens - 1):
            out_d = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out_d.past_key_values
            next_id = out_d.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    decode_ms = 1000 * (time.time() - t0)

    stop.set(); t.join()

    peak_vram = torch.cuda.max_memory_allocated(device) / 1e6
    avg_power = sum(samples) / max(1, len(samples))
    total_s = (prefill_ms + decode_ms) / 1000.0
    energy_J = avg_power * total_s
    decode_rate = (decode_tokens - 1) / (decode_ms / 1000.0) if decode_ms > 0 else 0

    return BenchRow(
        prompt_tokens=actual_prompt_tokens,
        decode_tokens=decode_tokens,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        decode_tok_per_s=decode_rate,
        peak_vram_mb=peak_vram,
        avg_power_w=avg_power,
        energy_J=energy_J,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="research_dev/llm_server/bench_gemma_a6000.json")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index (0 or 1)")
    ap.add_argument("--prompt_lens", type=int, nargs="+",
                    default=[1, 32, 128, 512, 1024])
    ap.add_argument("--decode_tokens", type=int, default=50)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")
    print(f"  dtype:  {next(model.parameters()).dtype}")

    # Warm-up
    print("\nwarm-up...")
    _ = run_one(model, tokenizer, device, prompt_tokens=32,
                decode_tokens=8, gpu_index=args.gpu)

    rows: list[BenchRow] = []
    for n in args.prompt_lens:
        print(f"\nbenchmarking prompt_tokens={n} ...")
        r = run_one(model, tokenizer, device, prompt_tokens=n,
                    decode_tokens=args.decode_tokens, gpu_index=args.gpu)
        rows.append(r)
        print(f"  prefill: {r.prefill_ms:.1f} ms   decode: {r.decode_ms:.1f} ms "
              f"({r.decode_tok_per_s:.1f} tok/s)   "
              f"peak VRAM: {r.peak_vram_mb:.0f} MB   "
              f"avg power: {r.avg_power_w:.1f} W   E: {r.energy_J:.1f} J")

    print("\n=== summary ===")
    print(f"{'prompt':>6} {'prefill_ms':>10} {'decode_ms':>9} {'tok/s':>8} "
          f"{'energy_J':>9} {'power_W':>7}")
    for r in rows:
        print(f"{r.prompt_tokens:>6} {r.prefill_ms:>10.1f} {r.decode_ms:>9.1f} "
              f"{r.decode_tok_per_s:>8.1f} {r.energy_J:>9.1f} {r.avg_power_w:>7.1f}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model_id": MODEL_ID, "device": device, "rows": [asdict(r) for r in rows]},
        indent=2,
    ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
