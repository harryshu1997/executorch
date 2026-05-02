"""
Benchmark vJEPA2 ViT-L on A6000 server tier — gives us the actual offload
speedup number vs OP15 Adreno GPU (~11 s/inf at fpc2 fp16).

Times the same encoder forward pass we measured on phone, both fp32 and
fp16, with bf16 mixed precision. Reports tok/s, peak VRAM, avg power.
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
from transformers import AutoModel

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


@dataclass
class Row:
    frames: int
    dtype: str
    warm_ms: float
    avg_ms: float
    peak_vram_mb: float
    avg_power_w: float
    energy_J_per_inf: float


def gpu_power_sampler(stop_event, samples, gpu_index=0):
    while not stop_event.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--id={gpu_index}",
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            samples.append(float(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.5)


def time_one(model, frames, dtype, device, n_iter=5, gpu_index=0):
    H = W = 256
    x = torch.zeros(1, frames, 3, H, W, dtype=dtype, device=device)

    samples: list[float] = []
    stop = threading.Event()
    t = threading.Thread(target=gpu_power_sampler, args=(stop, samples, gpu_index))
    t.start()

    torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model.encoder(x).last_hidden_state
    torch.cuda.synchronize()
    warm_ms = 1000 * (time.time() - t0)

    # Steady-state
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(n_iter):
            _ = model.encoder(x).last_hidden_state
    torch.cuda.synchronize()
    avg_ms = 1000 * (time.time() - t0) / n_iter

    stop.set(); t.join()
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e6
    avg_power = sum(samples) / max(1, len(samples))
    energy_J = avg_power * (avg_ms / 1000.0)

    return Row(frames=frames, dtype=str(dtype),
               warm_ms=warm_ms, avg_ms=avg_ms,
               peak_vram_mb=peak_vram, avg_power_w=avg_power,
               energy_J_per_inf=energy_J)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_json", default="research_dev/vjepa2/bench_vjepa2_a6000.json")
    ap.add_argument("--frames", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"loading {MODEL_ID} on {device}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")

    rows: list[Row] = []
    for dtype in (torch.float32, torch.float16):
        m = model.to(dtype) if dtype is not torch.float32 else model
        for f in args.frames:
            try:
                r = time_one(m, f, dtype, device, n_iter=5, gpu_index=args.gpu)
                rows.append(r)
                print(f"  fpc{f:>2}  {str(dtype).split('.')[-1]:<7} "
                      f"warm={r.warm_ms:>7.1f} ms  "
                      f"avg={r.avg_ms:>7.1f} ms  "
                      f"vram={r.peak_vram_mb:>5.0f} MB  "
                      f"power={r.avg_power_w:>4.0f} W  "
                      f"E={r.energy_J_per_inf:>5.1f} J")
            except Exception as e:
                print(f"  fpc{f:>2}  {dtype}: {e}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model_id": MODEL_ID, "device": device, "rows": [asdict(r) for r in rows]},
        indent=2,
    ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
