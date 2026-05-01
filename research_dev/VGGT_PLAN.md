# Collaborative Edge–Phone–Cloud 3D Reconstruction Prototype Plan (VGGT)

A phased plan to build a research prototype that demonstrates energy savings for
continuous 3D point-cloud reconstruction from egocentric video by splitting
VGGT's compute across a battery-limited "glasses" proxy, a phone, and a local
server. The goal is a publishable Pareto curve (reconstruction quality vs. total
Wh) with a concrete AR/VR motivation — not a production system.

VGGT repo: [/home/myid/zs89458/Documents/vggt](../../../vggt). Model:
[vggt/models/vggt.py](../../../vggt/vggt/models/vggt.py),
aggregator/encoder: [vggt/models/aggregator.py](../../../vggt/vggt/models/aggregator.py).
Heads: [vggt/heads/](../../../vggt/vggt/heads/).

---

## 0. Why VGGT, why this application

**Target application**: AR/VR glasses continuously stream egocentric video. A
scene is progressively reconstructed as a 3D point cloud; the headset-local
budget is ~1–2 Wh total while worn. Full cloud offload wastes radio energy on
raw frames; full on-device 3D is infeasible on a Hexagon-class DSP. Splitting
the model *inside* VGGT is the interesting question.

**Why VGGT over DUSt3R / MASt3R / Spann3R**:

- CVPR 2025 SOTA. Current reference that almost every recent paper benchmarks
  against.
- Encoder is DINOv2-L — already well-studied for quantization (contrasted with
  CroCo v2 in DUSt3R).
- Factored architecture — aggregator + independent heads — is naturally
  split-friendly.
- Commercial-licensed checkpoint ([VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial))
  removes downstream release friction.

---

## 1. Success criteria (define upfront)

- End-to-end run of a ScanNet++ walkthrough (50+ frames) through glasses-proxy →
  phone → server producing a point cloud.
- **Measured** (not modeled) energy per tier, via the perfetto-based harness we
  already have ([`run_video_on_device.py`](edge_encoder/run_video_on_device.py))
  within ±10% of repeated runs.
- Pareto plot with ≥5 configurations: cloud-only, phone-encoder-only,
  glasses+phone+server split, + 2 sampling/compression variants.
- Reconstruction quality within a target Chamfer-distance gap (e.g., ≤20% worse
  than cloud-only baseline) at the "efficient" endpoint.

---

## 2. Architecture: where to split VGGT

VGGT's forward ([vggt/models/vggt.py:28-94](../../../vggt/vggt/models/vggt.py)):

```
images (B,S,3,518,518)
   ↓ [Aggregator — 24 alternating-attention blocks]
aggregated_tokens_list, patch_start_idx
   ↓ [4 heads, all independent]
camera pose, depth + conf, world points + conf, tracks
```

**Aggregator internals** ([vggt/models/aggregator.py](../../../vggt/vggt/models/aggregator.py)):

- `patch_embed = "dinov2_vitl14_reg"` — DINOv2-L + 4 register tokens.
- `aa_order=["frame", "global"], aa_block_size=1` — each of the 24 blocks
  alternates between **frame self-attention** (per-frame, S-independent) and
  **global cross-attention** (across all S frames).

**Split strategy** (three candidate boundaries):

| Split | Glasses (OP11) | Phone (OP12) | Server (A6000) | Notes |
|---|---|---|---|---|
| A. Frame-only | — | patch_embed + all frame-attn blocks only | all global-attn blocks + 4 heads | Requires refactoring aggregator into two callable halves. Clean theoretically; moderate code. |
| B. Encoder/decoder | — | full aggregator | 4 heads | Mirrors what we did for LLaVA (encoder on phone, head on server). Easiest starting point. |
| C. Three-tier | DINOv2 patch embed + first K blocks | remaining frame-attn blocks | global-attn + heads | Most ambitious; gives glasses-tier Pareto point. |

**Recommendation**: start with **B** (encoder on phone, heads on server) because
we already have the export pipeline working. Add **A** as the Phase-5 extension
(frame-only tokens are smaller to upload than full aggregator output, giving a
concrete network-energy win). **C** is optional — only if OP11 gives
meaningfully different numbers than OP12 and there's time.

---

## 3. Stack per tier

| Tier | Hardware | Framework | Role |
|---|---|---|---|
| "Glasses" proxy | OnePlus 11 (SM8550) | **ExecuTorch + QNN (HTP fp16 or 16a8w)** | DINOv2 patch embed + first K frame-attn blocks. Optional in phase 6. |
| Phone | OnePlus 12 (SM8650) | **ExecuTorch + QNN (HTP fp16)** | VGGT aggregator (or frame-attn portion). Uploads tokens to server. |
| Local server | RTX A6000 | **Python + FastAPI + PyTorch** | Global-attn blocks, all heads, point-cloud fusion, optional gaussian-splatting post-processing. |
| Benchmark sink | (same A6000 or separate) | PyTorch | Cloud-only reference for Pareto baseline. |

No cloud GPU needed for core experiments. VGGT is feed-forward and fits on a
single A6000.

---

## 4. Dataset

**Primary: [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)** — 460 indoor
scenes with high-quality 3D mesh ground truth via Faro scanner. Used by VGGT's
own eval. Free for research; access via 2-minute form.

**Fallback / fast-iteration: Replica** — 18 small indoor scenes, sub-GB
download, used by most depth/reconstruction papers as a cheap secondary.

Sequence handling: sample 50–200 frames per scene at target fps (e.g., 2 fps
camera path), feed through the pipeline, reconstruct a single fused point cloud
per scene, measure against GT mesh.

---

## 5. Phases

Ordered by dependency, not time. Exit each phase only when its deliverable is
reproducible — same rule as the VLM plan.

### Phase 1 — Cloud baseline + harness

**Goal**: reproduce published VGGT numbers and establish the
Wh-per-reconstruction baseline everything else is measured against.

- Install VGGT dependencies; load `facebook/VGGT-1B` (or `VGGT-1B-Commercial`
  after the application approval).
- Run the bundled [`demo_colmap.py`](../../../vggt/demo_colmap.py) on a single
  Replica scene end-to-end to confirm the environment works.
- Download a 10-scene ScanNet++ subset. Run VGGT cloud-only (images uploaded
  → A6000 → point cloud) and log `nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv`
  at 1 Hz.
- Compute Chamfer distance and completeness vs ScanNet++ GT for each scene.
- Commit: `baselines/cloud_only/` with per-scene metrics + energy.

**Exit criterion**: cloud-only mean Chamfer within ±5% of VGGT's published
ScanNet++ numbers; Wh/scene logged and reproducible over 3 runs.

### Phase 2 — Encoder extraction + phone export (split B)

**Goal**: a `.pte` that produces aggregator output numerically close to the
cloud.

- Extract `VGGT.aggregator` into a standalone `nn.Module` whose `forward` takes
  `(S, 3, 518, 518)` and returns the aggregator's `(aggregated_tokens_list,
  patch_start_idx)`. This mirrors [`VisualOnly`](edge_encoder/llava/export_qnn.py)
  from the LLaVA work.
- Apply the same export recipe we used for LLaVA: `torch.export.export` →
  `to_edge_transform_and_lower_to_qnn` (fp16, SM8650).
- Gotchas to watch for:
  - **Sequence length S is dynamic** in VGGT. Pick a canonical S (e.g., 8 or 16)
    and export a static-shape `.pte` per canonical S — same trade-off we
    accepted for Qwen2-VL.
  - **Rotary pos embedding** — `RotaryPositionEmbedding2D` in [aggregator.py](../../../vggt/vggt/models/aggregator.py)
    may have data-dependent shape logic. If it breaks export, bake the position
    embedding as a buffer computed outside forward — same pattern as Qwen2-VL's
    `VisualOnly`.
  - **Frame↔global attention alternation** — Block routes through different
    code paths per block. Verify export traces both paths.

**Exit criterion**: eager fp16-cuda vs exported `.pte` cosine similarity ≥0.99
on the output of aggregator block 24, on a canonical 8-frame input.

### Phase 3 — End-to-end split pipeline (phone encoder → server heads)

**Goal**: a working pipeline where OP12 computes aggregator features and A6000
runs the heads, producing a point cloud equivalent to cloud-only.

- On OP12: push `.pte` + pre-processed frame `.raw` files, run via existing
  `qnn_executor_runner`. Output is aggregator tokens (~MB-per-frame depending
  on S and embed_dim).
- On A6000: a FastAPI endpoint that receives aggregator tokens, injects them
  into a VGGT model with the encoder bypassed, runs the heads, and emits a
  point cloud.
- Numerical validation: phone-encoded + server-decoded point cloud vs
  cloud-only point cloud — per-scene Chamfer + per-point cosine check at the
  pre-head feature level.
- Energy: reuse [`run_video_on_device.py`](edge_encoder/run_video_on_device.py)
  with the perfetto charge_uah path for OP12; A6000 side use `nvidia-smi`
  integration with network bytes counted.

**Exit criterion**: split reconstruction Chamfer within 10% of cloud-only;
phone ViT + network upload total Wh < cloud-only-with-raw-upload total Wh on a
mobile-network model (0.117 kWh/GB).

### Phase 4 — Frame selection

**Goal**: cut energy by not uploading every frame.

- Compare 3 policies:
  1. Uniform subsample every Nth frame.
  2. **Motion-based** — frame-to-frame image difference (pHash or optical-flow
     magnitude) above threshold.
  3. **Aggregator-feature-based** — phone-side cosine between consecutive
     aggregator outputs; skip if similarity > τ. Requires the aggregator output
     to be useful pre-global-attention, which **split B** gives us for free.
- Sweep fps-equivalent rates (30, 15, 6, 2, 1 fps). Plot Chamfer vs Wh.

**Exit criterion**: at 1/10th frames uploaded vs Phase 3 baseline, Chamfer
degrades by less than 2× (acceptable trade for paper story).

### Phase 5 — Token compression

**Goal**: shrink each uploaded token payload to cut network energy.

- VQ-KMeans (sklearn, 65k or 256k codebook) on aggregator features across
  Phase-1 training scenes. Implemented in
  `orchestrator/compress.py` (to be written).
- Optional INT8 post-quantize of aggregator tokens before upload (this is pure
  storage quantization, not model quantization — doesn't affect the on-device
  compute path).
- Alternative: **frame-only token split** (design A) — upload only
  frame-attention output, let server run global attention. Smaller tokens
  because global cross-attention hasn't expanded the representation yet.

**Exit criterion**: ≥5× reduction in bytes-per-frame with <1% Chamfer
degradation.

### Phase 6 — Three-tier (glasses + phone + server)

**Goal**: an optional Pareto point showing glasses-tier participation.

- Run patch_embed + first K frame-attention blocks on OP11 (SM8550). Output:
  per-frame DINOv2-L features (+ register tokens).
- Phone pulls those over local HTTP (LAN), runs the remaining encoder blocks
  + alternating attention, uploads result to server.
- Measurement: three perfetto sessions in parallel, one per device.

**Exit criterion**: glasses-tier reduces per-frame phone energy by ≥30% at
equivalent Chamfer quality. (If this fails, drop back to 2-tier; not
paper-critical.)

### Phase 7 — Evaluation sweep

**Goal**: the Pareto plot.

- Benchmark: 20–50 ScanNet++ scenes (subset of the official test split).
- Configs: cloud-only, 2-tier fp16, 2-tier fp16 + frame-skip, 2-tier + VQ,
  3-tier, 3-tier + frame-skip + VQ.
- Axes: **total Wh per scene** (edge + phone + server + network) vs **Chamfer
  distance**. Error bars from 3 repeats per config.

**Exit criterion**: ≥5 configs on one chart; "efficient" endpoint reaches the
success-criterion quality gap (≤20%) at <30% of the cloud-only Wh.

### Phase 8 — Writeup & release

**Goal**: external artifact. HotMobile, MLSys workshops, or a NeurIPS
efficiency workshop track are the right venues for this compute-split paper.

---

## 6. Measurement plan

| Tier | Tool | Notes |
|---|---|---|
| Phone (OP12) | Perfetto `android.power` + `batt.charge_uah` at 4 Hz via [`run_video_on_device.py`](edge_encoder/run_video_on_device.py) | Validated to ~1 µAh resolution on OP12; no root needed |
| Glasses (OP11) | Same perfetto harness | Assumes OP11 has the same unlocked perfetto data sources |
| Server | `nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=100` | Integrate W·s over inference windows |
| Network | Modeled at **0.117 kWh/GB mobile, 0.03 kWh/GB fixed** | Same convention as the VLM plan |
| Total | `energy/aggregate.py` | Sum per-scene across tiers |

The single biggest sanity check: sum of per-tier Wh should track total battery
drain on each device within 15%.

---

## 7. Evaluation matrix

Minimum 2D Pareto:

| Config | Chamfer (cm ↓) | Total Wh ↓ |
|---|---|---|
| Cloud-only baseline | X₀ | Y₀ |
| 2-tier fp16 | | |
| 2-tier + frame-skip (1 fps) | | |
| 2-tier + VQ tokens (256k codebook) | | |
| 2-tier + frame-skip + VQ | | |
| 3-tier (glasses + phone + server) | | |
| 3-tier + everything | | |

Target: "everything" row within 20% of cloud-only Chamfer at <30% of Wh.

---

## 8. Risks & mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| VGGT aggregator export fails due to alternating-attention control flow | Medium | Refactor aggregator into two static submodules (`frame_block`, `global_block`); export each separately. If still blocked: export only the frame-attn portion (matches design A anyway). |
| RoPE 2D shape logic has data-dependent ops | Medium | Same pattern as Qwen2-VL — precompute cos/sin as buffers based on canonical S. |
| DINOv2-L encoder outputs don't quantize to 8a8w cleanly (register tokens are known outlier sources) | High | Stay fp16 for edge; note 8a8w as future work. Same outcome we accepted for LLaVA-CLIP. |
| ScanNet++ access delayed | Low | Start with Replica (public, instant). Swap in ScanNet++ when access arrives. |
| Phone-to-server network throughput (~100 Mbps WiFi) becomes the bottleneck | Medium | Profile actual bytes/sec per config; if network-bound, frame-skip and VQ (Phases 4–5) become load-bearing. |
| 3-tier OP11+OP12 clock drift desynchronizes tiers | Low | NTP both phones; align perfetto timestamps post-hoc using `adb shell date`. |

---

## 9. Repo structure (extends the existing work)

```
research_dev/
├── VGGT_PLAN.md                    # this file
├── PROTOTYPE_PLAN.md               # prior VLM plan (preserved)
├── edge_encoder/                   # LLaVA/Qwen2-VL pipeline (reused infra)
│   ├── run_video_on_device.py      # perfetto energy harness — REUSED
│   └── sample_video_frames.py      # per-model preprocess — extend for VGGT
└── vggt_split/                     # NEW
    ├── cloud_baseline/
    │   ├── run_scannetpp.py        # Phase 1
    │   └── chamfer.py              # shared metric
    ├── edge/
    │   ├── extract_aggregator.py   # pulls Aggregator out of VGGT, wraps for export
    │   ├── export_qnn.py           # exports phone .pte
    │   └── phase6_glasses.py       # OP11 first-K-blocks .pte (optional)
    ├── server/
    │   ├── serve_heads.py          # FastAPI endpoint receiving aggregator tokens
    │   └── fuse_pointclouds.py     # multi-frame fusion
    ├── compression/
    │   ├── vq_codebook.py          # fit + apply
    │   └── frame_selector.py       # motion / feature-based policies
    ├── energy/
    │   ├── collect_phone.py        # reuse harness
    │   ├── collect_gpu.py          # nvidia-smi sampler
    │   └── aggregate.py            # sum tiers + network
    └── eval/
        ├── run_pareto.py           # driver for the full sweep
        └── plot_pareto.py
```

---

## 10. Infrastructure reuse from the VLM work

Already built, directly usable:
- **ExecuTorch + QNN SDK + NDK + ADB** — environment set up, verified by
  inception_v4 and LLaVA runs.
- **Perfetto-based energy harness** — [`run_video_on_device.py`](edge_encoder/run_video_on_device.py),
  charge_uah at 4 Hz, no root, units fixed.
- **Idle baseline**: 327 mW on OP12 (screen-off, WiFi-on). Use the same
  subtraction method.
- **Export patterns**: wrapper module, `torch.export(..., strict=False)`,
  `to_edge_transform_and_lower_to_qnn`, compile for SM8650.
- **Calibration utilities** — COCO cache, image-processor wrapping,
  `make_quantizer` recipe. Less relevant if we stay at fp16 for VGGT, but
  available if we attempt 16a8w in a later phase.

What's new (genuinely):
- VGGT codebase setup; checkpoint access (commercial or research).
- Aggregator extraction and the "two-half" module pattern (split A).
- Head-serving FastAPI.
- Multi-view point-cloud fusion.
- ScanNet++ benchmark harness + Chamfer metric.
- Frame selection policies.
- VQ codebook training.

---

## 11. Budget

- **Hardware**: OP12 + OP11 + A6000 you already have. ScanNet++ is free.
- **Cloud**: none for core experiments; optional A100 cloud run only if a
  "further offload" Pareto point is needed (~$20).
- **Software**: free (VGGT, PyTorch, ExecuTorch, Perfetto).
- **Time**: ~160 h focused work (20 h/week for 8 weeks) on top of the already-
  complete VLM infrastructure.

---

## 12. First-week checklist

1. `cd /home/myid/zs89458/Documents/vggt && pip install -e . && pip install -r requirements.txt`
2. `python demo_colmap.py` on one Replica sequence — smoke test for VGGT on
   A6000. If this fails, stop and debug before touching the split.
3. Fill out ScanNet++ access form; download Replica in parallel so Phase 1 can
   start immediately.
4. Load `VGGT` in Python, print `model.aggregator` module tree to understand
   how to subclass/slice it for export (same drill we did for LLaVA's CLIP).
5. Run `model(images[:8])` with a synthetic `(8, 3, 518, 518)` input on A6000;
   time it and record baseline ms/frame — this is the cloud-only latency we
   compare against.
6. If step 2 fails at step 4 ("aggregator slicing is complex"), file a VGGT
   GitHub issue or work around with a `forward_until_block(k)` patch.

Every step above should take ≤1 day. If step 4 or 5 takes more than 2 days,
the Phase-2 export scope is a larger risk than currently rated.
