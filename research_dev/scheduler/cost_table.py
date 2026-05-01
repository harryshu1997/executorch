"""
(model, device, concurrent_set) -> (latency_ms, energy_J, output_bytes)

Initial values mix real measurements we already took with clearly-marked
placeholders. The simulator uses whatever is here; Phase V/A/L of
MULTI_MODEL_PLAN replaces placeholders one row at a time.

Units:
    latency_ms   : wall-clock on that device, includes model forward only
    energy_J     : device-side energy; *excludes* network transfer to next stage
    output_bytes : size of the result that must be shipped to its consumer
"""
from __future__ import annotations

from dataclasses import dataclass

# -- Network model (from MULTI_MODEL_PLAN §6) -----------------------------
# Joules per byte of transfer on each link class.
# Mobile radio:  0.117 kWh/GB = 4.212e-7 J/byte
# Fixed (wifi):  0.030 kWh/GB = 1.080e-7 J/byte
J_PER_BYTE_MOBILE = 0.117 * 1000 * 3600 / 1e9   # ≈ 4.21e-7
J_PER_BYTE_FIXED = 0.030 * 1000 * 3600 / 1e9   # ≈ 1.08e-7

# Crude per-byte latency model: 100 Mbps wifi / 20 Mbps mobile.
BYTES_PER_SEC_WIFI = 100e6 / 8   # ~12.5 MB/s
BYTES_PER_SEC_MOBILE = 20e6 / 8  # ~2.5 MB/s


@dataclass(frozen=True)
class Cost:
    latency_ms: float
    energy_J: float
    output_bytes: int
    note: str = ""          # provenance: "measured" / "estimate" / "placeholder"


# Data snapshot — edit freely as measurements come in.
# Key convention: (model_name, device_name).  Concurrent-set dimension
# is collapsed for now (solo-only); extend when we measure contention.
_TABLE: dict[tuple[str, str], Cost] = {
    # ---- Vision reconstruction: StreamVGGT (replaces vanilla VGGT) -----
    # MULTI_MODEL_PLAN §13f: vanilla VGGT's 67 MB/frame aggregator output
    # made every placement infeasible. StreamVGGT's causal-transformer
    # variant with KV-cache ships only 6.44 MB of head outputs per frame,
    # which is what the downstream LLM / scene-graph actually consumes.
    #
    # Trace events say "VGGT_encoder" but we now price the *full* StreamVGGT
    # pipeline (aggregator + heads) per arrival — head outputs are the
    # deliverable.
    #
    # Latency uses steady-state at S=16 (realistic for a 10-min AR session),
    # not the S=1 cold-start number. See research_dev/streamvggt/
    # profile_streamvggt.py output.
    #
    # Phone number is still an upper-bound estimate — StreamVGGT aggregator
    # has not yet been exported to QNN. KV cache on phone will dominate the
    # design once we get there (270 MB/frame growth in fp32; fp16 halves it).
    ("VGGT_encoder", "phone"):   Cost(1500, 8.0,  6_440_000, "estimate full StreamVGGT on OP12; needs QNN export"),
    ("VGGT_encoder", "server"):  Cost( 545, 60.0, 6_440_000, "measured A6000 fp32, S=16 steady-state streaming"),
    ("VGGT_encoder", "cloud"):   Cost( 400, 45.0, 6_440_000, "estimate: A100 fp16 ~1.4x faster"),

    # VGGT_heads: no longer a separate row — StreamVGGT integrates them
    # into the per-frame streaming call. Left as zero-cost stubs for any
    # trace that still references them, so existing traces don't break.
    ("VGGT_heads",   "server"):  Cost(  0,  0.0,          0, "absorbed into VGGT_encoder (StreamVGGT)"),
    ("VGGT_heads",   "cloud"):   Cost(  0,  0.0,          0, "absorbed into VGGT_encoder (StreamVGGT)"),

    # ---- LLaVA CLIP ViT (proxy for VGGT encoder pattern) ---------------
    # Real numbers from our OP12 run: 143 ms solo, 1.7 J per frame.
    ("LLaVA_vit",    "phone"):   Cost(143,  1.7,  9_437_184, "measured OP12 HTP fp16"),
    ("LLaVA_vit",    "server"):  Cost( 15,  3.0,  9_437_184, "estimate: A6000 fp16"),

    # ---- Qwen2-VL ViT --------------------------------------------------
    ("Qwen2VL_vit",  "phone"):   Cost(1909, 10.0, 5_605_376, "measured OP12 HTP 16a8w"),
    ("Qwen2VL_vit",  "server"):  Cost(  60, 12.0, 5_605_376, "estimate"),

    # ---- Whisper-Tiny --------------------------------------------------
    # Phone: MEASURED on OP15 (Hexagon v81 NPU) via qnn_executor_runner over
    # whisper_qnn_16a8w.pte produced from examples/qualcomm/oss_scripts/whisper.
    # 5-shot avg @ htp_performance_mode=4: encoder 105.5 ms + decoder 11.5 ms
    # ≈ 117 ms full pipeline per utterance (default 30 s audio, single decode
    # step). Energy is still an estimate pending perfetto charge_uah pass.
    ("Whisper",      "phone"):   Cost(117,  2.0,     20_000, "measured OP15 NPU 16a8w; energy est."),
    ("Whisper",      "server"):  Cost( 60,  4.0,     20_000, "placeholder"),
    ("Whisper",      "cloud"):   Cost( 80,  5.0,     20_000, "placeholder"),

    # ---- Small assistant LLM (phone-never) -----------------------------
    # 'phone' is intentionally absent: this is a fixed-strategy model.
    # Server: measured Gemma-4-E2B-it on A6000 bf16 — prefill 50 ms (avg over
    # 32/128/512 prompt-token lengths) + decode 50 tokens @ 27 tok/s = ~1850 ms
    # total. Energy: 230 J avg @ 120 W. Output: 50 tokens × ~5 bytes ≈ 250 B
    # (text), but realistic answer payload (JSON envelope + audio synth hint)
    # rounded to ~10 KB.
    ("LLM_query",    "server"):  Cost(1900, 230.0,    10_000, "measured Gemma-4-E2B-it A6000 bf16, 50-tok output"),
    ("LLM_query",    "cloud"):   Cost(1400, 165.0,    10_000, "estimate: Gemma-4-E2B on A100 bf16 (~1.4x faster)"),
}


# Which devices can run which models (derived from _TABLE keys).
_COMPATIBILITY: dict[str, list[str]] = {}
for (model, device), _ in _TABLE.items():
    _COMPATIBILITY.setdefault(model, []).append(device)


class CostTable:
    """Thin facade so simulator code doesn't touch the dict directly."""

    def get(self, model: str, device: str) -> Cost | None:
        return _TABLE.get((model, device))

    def devices_for(self, model: str) -> list[str]:
        return list(_COMPATIBILITY.get(model, ()))

    def all_models(self) -> set[str]:
        return set(_COMPATIBILITY.keys())


def network_cost(bytes_: int, link: str) -> tuple[float, float]:
    """Returns (latency_s, energy_J) for moving `bytes_` over `link`.

    `link`:
        "phone->server" (wifi / fixed)
        "phone->cloud"  (mobile)
        "server->cloud" (fixed)
    """
    if link == "phone->server":
        return (bytes_ / BYTES_PER_SEC_WIFI, bytes_ * J_PER_BYTE_FIXED)
    if link == "phone->cloud":
        return (bytes_ / BYTES_PER_SEC_MOBILE, bytes_ * J_PER_BYTE_MOBILE)
    if link == "server->cloud":
        return (bytes_ / BYTES_PER_SEC_WIFI, bytes_ * J_PER_BYTE_FIXED)
    return (0.0, 0.0)  # same-device
