# Collaborative Edge–Cloud Video VLM Prototype Plan

A phased plan to build a research prototype that demonstrates energy savings for
long-form video understanding by collaboratively distributing a VLM pipeline across
two phones (OnePlus 12, OnePlus 11), one desktop (RTX 4060 Ti), and one cloud GPU
(A100). The goal is a publishable Pareto curve (accuracy vs energy), not a
production system.

---

## 0. Success criteria (define upfront)

- End-to-end run of one video through phones → desktop → cloud, producing a correct
  VideoMME-style answer.
- **Measured** (not modeled) energy for each tier, within ±10% of the MLCarbon
  prediction.
- Pareto plot with ≥5 configurations (cloud-only, single-phone, two-phone,
  two-phone + merging, two-phone + merging + filtering).
- Accuracy within 3 points of cloud-only baseline on a held-out benchmark subset.

---

## 1. Stack per tier

| Tier | Framework | Role |
|---|---|---|
| Phones (OP12, OP11) | **ExecuTorch + QNN backend** | ViT forward (INT8 on Hexagon) + frame filter |
| Desktop (4060 Ti) | **Python + FastAPI + PyTorch** | Orchestration, token VQ/merging, upload, optional small local VLM |
| Cloud (A100) | **vLLM** (or HF Transformers) serving **Qwen2-VL-7B** | LLM prefill + decode on pre-computed embeddings |

---

## 2. Phases

Phases are ordered by dependency, not time. Exit each phase only when its
deliverable is reproducible end-to-end — otherwise the next phase will paper over
a bug you'll have to debug later.

### Phase 1 — Foundation: environment + cloud baseline

**Goal**: a known-good toolchain on both ends plus a ground-truth energy number
to compare everything against.

- Install ExecuTorch (main branch), Qualcomm QNN SDK ≥2.25, Android Studio, Android NDK.
- Enable ADB + developer mode on at least one phone; verify `adb shell` works.
- **Baseline measurement**: run Qwen2-VL-7B on a rented A100 (Lambda/RunPod, ~$2/hr)
  for 10 VideoMME clips. Log `nvidia-smi -lms 1000`. Compute actual IT kWh → ground
  truth for all later comparisons.
- Commit: `baselines/cloud_only/` with energy logs.

**Exit criterion**: cloud baseline accuracy + energy are logged and reproducible.

### Phase 2 — Export Qwen2-VL's ViT to ExecuTorch

**Goal**: a `.pte` file that produces embeddings numerically close to the cloud ViT.

- Load Qwen2-VL-7B from HuggingFace; extract **just** the vision encoder + patch merger.
- `torch.export` → `to_edge` → `to_backend(QnnPartitioner)` → `.pte` file.
- First try **fp16** on HTP to confirm the pipeline runs end-to-end. Get one image through.
- Then add **pt2e INT8 quantization** with ~200 calibration images (ImageNet subset works).

**Exit criterion**: cosine similarity between on-device embeddings and FP32 cloud
embeddings >0.98 on a 100-image check set.

### Phase 3 — Single-phone Android app

**Goal**: a phone app that turns a video into an embeddings file, with real energy
measurements.

- Minimal Kotlin app: pick video file → `MediaMetadataRetriever` to sample frames →
  call `.pte` via ExecuTorch Android binding → write embeddings to disk.
- Measure energy: `adb shell dumpsys batterystats --reset`, run, dump, parse with
  **Battery Historian**. Cross-check with a USB power meter (e.g., YZXStudio) if
  available — Battery Historian alone undercounts NPU draw by 10–30%.

**Exit criterion**: "video → embeddings file on phone, X Wh consumed, Y minutes
wall-clock" reported deterministically across three runs.

### Phase 4 — Scene filter + two-phone split

**Goal**: two phones working in parallel with per-phone frame budgets.

- Add lightweight filter: frame-diff hashing (pHash) or MobileNetV3 embedding
  similarity. Keep it on-device.
- Port app to OP11 (same `.pte` works on both — both Snapdragon/Hexagon).
- Simple work split: alternating chunks, or proportional (OP12 gets 63%, OP11 gets 37%).
- Phones write embeddings to desktop over local HTTP.

**Exit criterion**: both phones process a video in parallel; combined wall-clock
is roughly half the single-phone time.

### Phase 5 — Desktop orchestrator

**Goal**: a single endpoint on the desktop that accepts phone uploads, compresses,
and forwards.

- FastAPI service: receives embeddings from phones, reorders, runs VQ quantization
  (sklearn `KMeans` codebook with 65k entries, or just int8 rounding as a first pass).
- Optional token merging (2×2 spatial pool of adjacent-frame tokens).
- Uploads to cloud endpoint.
- Log CPU/GPU power via `nvidia-smi` + `powercap-info` (RAPL).

**Exit criterion**: a video uploaded to phones ends up as a compressed token
blob on the cloud endpoint, with desktop energy measured.

### Phase 6 — Cloud service accepting pre-computed embeddings

**Goal**: an LLM prefill+decode path that skips the ViT.

- **Key hack**: vLLM doesn't cleanly accept `inputs_embeds` for Qwen2-VL out of the box.
  Two options:
  - **A (easier)**: use HF `transformers` directly with
    `model.generate(inputs_embeds=..., ...)`. Slower than vLLM but ~3× faster than
    baseline because you skipped the ViT.
  - **B (faster)**: patch vLLM's Qwen2-VL forward to accept pre-computed visual
    features. ~1 day of work if you're comfortable in vLLM internals.
- Run on rented A100; log `nvidia-smi`.

**Exit criterion**: a token blob → correct VideoMME-style answer, with cloud
energy measured per request.

### Phase 7 — Evaluation sweep

**Goal**: the Pareto plot that drives the paper.

- Pick a benchmark subset: **100 VideoMME clips** (short + medium) is enough for
  a prototype.
- Sweep configs: {cloud-only, single-phone, two-phone, +merging, +filtering, +everything}.
- For each: measure (phone Wh + desktop Wh + cloud kWh + network kWh) and accuracy.
- Plot **accuracy vs total Joules** — this is the figure that sells the paper.

**Exit criterion**: ≥5 configs on one chart, with error bars from 3 repeats.

### Phase 8 — Writeup & release

**Goal**: an external artifact.

- Extended abstract for **HotCarbon** (deadline usually late spring) or workshop
  track at **MLSys** / **SenSys**.
- Open-source the repo.

**Exit criterion**: submission + public repo link.

---

## 3. Measurement plan

| Tier | Tool | Notes |
|---|---|---|
| Phone | **Battery Historian** + `batterystats` | Reset → run → dump. Per-process NPU wake energy is reported. |
| Phone (ground truth) | USB power meter inline with charger | Most accurate; the phone must run on charger, not battery, for repeatable numbers. |
| Desktop CPU | Linux RAPL (`/sys/class/powercap/...`) | Per-socket package power. |
| Desktop GPU | `nvidia-smi --query-gpu=power.draw` | 1 Hz polling is fine. |
| Cloud | `nvidia-smi` on the VM | Apply PUE × grid intensity for CO₂. |
| Network | **Model it** — 0.03 kWh/GB fixed, 0.117 mobile | You can't measure ISP energy directly. Note the uncertainty. |

---

## 4. Evaluation matrix

Minimum 2D Pareto grid:

| Config | VideoMME acc. (%) | Total energy (Wh) |
|---|---|---|
| Cloud-only baseline | X | Y |
| + single phone encode |  |  |
| + two phone encode |  |  |
| + scene filter (2 fps) |  |  |
| + token merge 4× |  |  |
| + everything |  |  |

Target: the "everything" row ≥95% of baseline accuracy at <10% of baseline energy.

---

## 5. Risks & mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Qwen2-VL ViT fails QNN INT8 quantization | **High** | Fallback 1: fp16 on HTP (still 3× more efficient than cloud). Fallback 2: swap to CLIP-ViT-L/14 (proven quantizable) and accept a retraining step. |
| Accuracy drops >5 pts with merging + filter | Medium | Keep one "quality" tier without merging; ablate what's costing you accuracy. |
| vLLM embedding injection too hard | Medium | Fall back to HF Transformers + batching. Slower but correct. |
| Phone thermal throttle kills sustained throughput | High | Run on charger with case off; split work into 20-min bursts with 5-min cooldowns. |
| Multi-phone sync is flaky | Low | Start with sequential (one phone at a time); add parallelism only after single-phone works. |

---

## 6. Repo structure (suggested)

```
collab-video-vlm/
├── edge_encoder/
│   ├── export_vit.py          # Torch → .pte pipeline
│   ├── calibration/           # INT8 calibration data
│   └── android_app/           # Kotlin + ExecuTorch bindings
├── orchestrator/
│   ├── server.py              # FastAPI on desktop
│   ├── quantize.py            # VQ + token merging
│   └── scheduler.py           # work split across phones
├── cloud_server/
│   ├── serve_embeds.py        # HF / patched-vLLM endpoint
│   └── requirements.txt
├── energy/
│   ├── collect_phone.sh       # batterystats + historian
│   ├── collect_gpu.sh         # nvidia-smi sampler
│   └── aggregate.py           # merge logs → per-config Wh
├── eval/
│   ├── run_videomme.py        # accuracy harness
│   └── pareto_plot.py
└── README.md
```

---

## 7. Budget

- **Cloud GPU**: ~20 A100-hours × $2 = **~$40** total (plenty for baseline + eval).
- **Hardware**: OP12 + OP11 + desktop you already have. Optional USB power meter ~$30.
- **Time**: ~240 hours of focused work (30/week for 8 weeks).

---

## 8. First-week checklist (start here)

1. `git clone https://github.com/pytorch/executorch && ./install_requirements.sh`
2. Download Qualcomm QNN SDK, put `$QNN_SDK_ROOT` in your shell env.
3. Run `examples/qualcomm/scripts/inception_v4.py` as smoke test (ViT-adjacent
   model that's known to work on HTP).
4. If that works → try exporting Qwen2-VL's ViT in Week 2. If it doesn't → open a
   QNN backend issue first, that's your blocker.

If you get stuck at step 3, that's the "paper-killing" risk — fix it before
investing more.
