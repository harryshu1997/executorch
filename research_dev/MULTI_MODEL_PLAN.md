# Multi-Model Multi-Device Energy-Aware Scheduler

Research plan for an energy-minimizing scheduler that places heterogeneous
AI tasks (vision reconstruction, speech recognition, LLM queries, …) across a
heterogeneous device hierarchy (glasses → phone → local server → cloud) under
realistic AR/VR workload traces.

Supersedes the single-model scope of [VGGT_PLAN.md](VGGT_PLAN.md), which
remains the reference for the VGGT-specific pipeline work — VGGT is now the
first of several models characterized for the scheduler's cost tables.

---

## 0. Why this problem

Existing split-compute papers measure **one model in isolation**. Real AR/VR
devices never run one model at a time. Vision reconstruction, speech
recognition, assistant LLMs, tracking, and sensor fusion all contend for the
same HTP, HVX, memory bandwidth, and radio. The question that matters for
energy is not "should I offload VGGT?" — it's **"given the current mix of
live tasks, devices, battery states, and network conditions, where should
each task run to minimize Wh while meeting deadlines?"**

That question is:
- Specific: per-(model, device) cost has to be *measured*, not assumed.
- Heterogeneous: different models call for different offload strategies.
- Dynamic: battery state, thermal state, and network change during a session.

The paper contribution is the scheduler + the empirical cost model under
contention that drives it. The single-model experiments are its validation
fuel.

---

## 1. Problem formulation

**Task set** `T = {t₁, t₂, …}`. Each task has:
- A *model* (the AI module it runs: VGGT-encoder, VGGT-heads, Whisper, tiny-LLM…).
- A *trigger* (periodic @ fps; on voice-activity; on-demand query).
- A *deadline* `D_t` (soft — hard deadlines out of scope).
- A *priority* (background / foreground / user-facing).
- *Dependencies* (e.g., assistant-LLM depends on VGGT-encoder + Whisper
  outputs from the same time window).

**Device set** `D = {glasses (OP11), phone (OP12), local_server (A6000),
cloud (A100 rented)}`. Each has:
- A set of *models compiled for it* (some models never run there — e.g.,
  Llama-3-8B never on glasses).
- *Battery budget* (for glasses/phone) — finite.
- *Instantaneous power* `W(S, device)` under any subset `S ⊆ T` currently
  executing. **Crucially, this is NOT additive in S** — contention is the
  research question.
- *Thermal state* → sustained-vs-burst capability.

**Network links** `L ⊆ D × D` with bandwidth, latency, and energy/byte.

**Scheduler output**: for each task instance `t` and its arrival time,
decide a placement `x_t ∈ D` (possibly also *when* if latency tolerates
delay). Objective:

    minimize ∑ device_Wh(t)
    subject to  latency(t) ≤ D_t for all t,
                battery_budget satisfied on glasses/phone over session.

This is NP-hard in general (knapsack / assignment / precedence). The
research question is how well a *practical* policy does.

---

## 2. Per-model strategies (heterogeneous)

Not every model is offloadable the same way. The plan below captures the
first three models and their natural placement space.

| Model | Strategy summary | Fixed vs flexible |
|---|---|---|
| **VGGT encoder** | Split: runs on phone HTP or local server; output (tokens) forwarded to server for heads. | **Flexible** — scheduler decides phone vs server. |
| **VGGT heads** | Always server-side. 1 GB model, DPT decoders, point-cloud fusion. | **Fixed** (server or cloud). |
| **Whisper-Tiny / distil-Whisper** | Fully on-device (phone HTP) is viable — 39–166M params, INT8-friendly. Cloud fallback if phone HTP busy. | **Flexible** — phone vs server. |
| **Assistant LLM** (Phi-3 or Llama-3-8B) | Local server or cloud only. Phone sends a **compressed request** (token IDs + optional VGGT summary + Whisper transcript). Too large for phone. | **Fixed** (server or cloud). |
| **(later) Eye tracker** | Hard-realtime 90 fps. Glasses-only. | **Pinned** (not scheduled). |
| **(later) Voice-activity detection** | Hard-realtime, ~1 MB. Glasses or phone. | **Pinned**. |

Each model's strategy dictates the *decision variables* the scheduler
actually controls — only the flexible ones.

---

## 3. Workload traces (THIS IS PRIORITY ONE)

Without a realistic trace, the scheduler evaluation is synthetic and
uncompelling. Candidate sources in priority order:

### 3a. Aria Everyday Activities (Meta Project Aria)

- Recorded on actual AR glasses (Project Aria).
- Multi-modal by design: eye tracking, IMU, 3 cameras, 7-channel audio.
- ~100 h of real everyday activities, ~150 sequences.
- Public, under [Project Aria Tools](https://facebookresearch.github.io/projectaria_tools/).
- **This is the best-fit trace** — it's the closest to "real glasses workload" that exists publicly.

### 3b. Ego4D / EgoExo4D (fallback)

- 3,670 h of egocentric video + audio + narrations.
- Not recorded on glasses (on various camera rigs), but egocentric.
- Standardized benchmarks (moment queries, action recognition) can stand in
  as "assistant LLM query" events.
- Public, 2-minute access form.

### 3c. Synthesize from these sources

Ready-made traces don't exist with the exact task mix we need. We **derive**
task arrivals from the raw recordings:
- **VGGT trigger**: every Nth video frame at a target fps (1–5 fps).
- **Whisper trigger**: voice-activity-detection (Silero-VAD) on the audio
  stream; each utterance = one Whisper invocation.
- **LLM trigger**: intent-detection on Whisper output (simple keyword
  matcher or a tiny classifier); each detected query = one LLM invocation.
- Optionally: action-segment boundaries from the annotations → LLM
  "describe what just happened" triggers.

Output format: a timestamped JSON trace

```json
{"t": 12.34, "task": "VGGT_encoder", "frame": "001.jpg", "deadline_s": 0.5, "priority": "background"}
{"t": 12.40, "task": "Whisper", "clip": "utt_07.wav", "deadline_s": 0.5, "priority": "foreground"}
{"t": 13.12, "task": "LLM",    "input": {"transcript": "...", "context_tokens": "..."}, "deadline_s": 2.0, "priority": "foreground"}
```

### 3d. Workload-trace phases (do this before per-model work)

- **Phase W1**: obtain access to Aria data (or Ego4D).
- **Phase W2**: derive 3–5 "realistic session" traces (e.g., walking around
  kitchen, cooking, driving, social conversation, phone use). 10 min each.
- **Phase W3**: validate traces make sense — trigger rates plausible, task
  mix resembles actual AR usage.
- **Phase W4**: freeze v1 of trace format; commit under `research_dev/traces/`.

**Exit criterion**: ≥3 trace files, each ≥10 min, with at least 2 active
task types simultaneously > 30% of the timeline.

---

## 4. Per-(model, device) cost characterization

For each (model, device) pair compile the model, measure:
- **Latency (ms/inference)** — solo, and under each contention combination
  (e.g., VGGT + Whisper running on same device).
- **Energy (J/inference)** — via perfetto charge_uah on phone/glasses,
  `nvidia-smi` on server/cloud. Differencing methodology:
  `E_marginal(model | concurrent_set) = E(concurrent_set ∪ {model}) − E(concurrent_set)`.
- **Output bytes** → determines upload energy per execution.

This produces a (potentially large) cost table:

```
cost[(model, device, concurrent_set)] = (latency_ms, energy_J, output_bytes)
```

Scheduler consults this at runtime.

### 4a. Contention characterization (the hard part)

A single-device concurrent run of M models is 2^M combinations. For 3
models this is 8 subsets — feasible. For 6 models it's 64 — not. We
compress using two assumptions and *validate both empirically*:

1. **Pairwise approximation**: marginal cost of adding model *m* to set *S*
   depends only on *S*, not its composition. Validate by spot-checking 3+
   subsets of the same size.
2. **Resource-class grouping**: models that stress the same resource (HMX,
   HVX, memory bandwidth) contend; models that stress different resources
   compose. E.g., Whisper (heavy on MFCC/audio) and VGGT (heavy on vision
   matmul) may be less contended than two VGGTs.

If both assumptions hold to within 10%, the cost table is tractable. If
they fail, that failure is itself a paper result.

### 4b. Models to characterize (incremental roll-out)

Order:
1. **VGGT encoder / heads** — already have infrastructure ready.
2. **Whisper-Tiny (39M)** — ExecuTorch has [existing examples](../examples/whisper) to fork.
3. **Phi-3-mini or Llama-3.2-1B** — small LLM with request-response flow.
4. **(optional) Silero-VAD, small tracker, eye tracker** — pinned tasks, characterized for realism but not scheduled.

---

## 5. Baseline policies

To compare the scheduler against:

| Baseline | Policy |
|---|---|
| **always-local** | Every flexible task runs on the closest compute device (glasses → phone → server preference order). If it doesn't fit, drop to the next tier. |
| **always-cloud** | Every flexible task runs on the highest compute tier (cloud). Raw inputs uploaded. |
| **round-robin** | Flexible tasks rotate across devices in order, ignoring cost. |
| **Neurosurgeon-like** | Static per-model split based on network-vs-compute trade-off at the per-layer level (published heuristic, 2017). |
| **Oracle** | ILP solution of the scheduling problem given *full knowledge* of future arrivals. Upper-bound reference. |

The scheduler must **dominate** always-local, always-cloud, round-robin,
and Neurosurgeon on realistic traces. Closing the gap to the oracle is the
quality metric.

---

## 6. Scheduler design alternatives

No choice yet; candidates in order of complexity:

1. **Static ILP** (offline): given a full trace, solve the assignment with
   an integer program. Not deployable but provides an upper bound.
2. **Online greedy heuristic**: at each task arrival, pick the device that
   minimizes *current marginal energy given current device states* subject
   to deadline feasibility. Fast and reasonable.
3. **Online greedy + lookahead**: greedy but simulates the next few seconds
   of arrivals assuming a trigger-rate model.
4. **Contextual bandit / MDP / RL**: learns from observed costs and adapts
   to thermal throttling, network changes, battery decay. Most novel,
   hardest to evaluate honestly.

**Starting point**: implement (1) and (2), benchmark, and only add (3)/(4)
if the gap to the oracle is large enough to justify it.

---

## 7. Phases

### Phase W — Workload traces (START HERE)

Goal: locked, versioned, realistic task-arrival traces.

- W1. Get Aria or Ego4D access.
- W2. Derive 3–5 traces.
- W3. Sanity-check trigger rates.
- W4. Commit `research_dev/traces/`.

**Exit**: see §3d above.

### Phase V — VGGT as the first model

Goal: reuse existing split-compute VGGT pipeline (per [VGGT_PLAN.md](VGGT_PLAN.md))
but populate a cost table, not build the complete paper.

- V1. Follow VGGT_PLAN.md through its Phase 3 (split working end-to-end).
- V2. For each (device ∈ {phone, server, cloud}), measure solo VGGT
  encoder latency + energy.
- V3. Populate the first rows of `cost[(VGGT_encoder, *, {})]`.

**Exit**: VGGT's row in the cost table is committed.

### Phase A — Audio: Whisper

Goal: second model in the cost table.

- A1. Export Whisper-Tiny to QNN (fork the ExecuTorch example).
- A2. Solo phone benchmark (latency / energy / output size).
- A3. Cloud (A6000) reference for same clips.
- A4. Add `cost[(Whisper, *, {})]` rows.

**Exit**: Whisper in the cost table.

### Phase L — LLM: small assistant

Goal: request-response flow that glasses/phone never execute.

- L1. Pick a model (Phi-3-mini or Llama-3.2-1B).
- L2. Serve on A6000 with a request protocol (query + optional context).
- L3. Measure response latency and server-side energy per query.
- L4. Measure upload bytes + upload-energy model.
- L5. Add rows to cost table.

**Exit**: LLM in the cost table.

### Phase C — Contention characterization

Goal: fill the off-diagonal cost table entries.

- C1. For each device, run all pairwise and triplewise combinations of
  {VGGT_enc, Whisper, LLM-request} → measure marginal cost.
- C2. Validate pairwise approximation (§4a).
- C3. Report the contention matrix as a standalone result.

**Exit**: contention matrix published; either assumption validated or its
failure documented.

### Phase S — Scheduler implementation

- S1. Implement ILP baseline (Gurobi or PuLP).
- S2. Implement online greedy.
- S3. Implement always-local / always-cloud / round-robin / Neurosurgeon
  baselines.
- S4. Wire to the cost table from Phases V/A/L/C.

**Exit**: scheduler.py runs end-to-end on a trace, produces a placement
sequence.

### Phase E — End-to-end evaluation

- E1. Run each policy against each trace. Execute the placement sequence
  on real hardware. Measure total Wh.
- E2. Compute per-policy stats: energy, deadline-miss rate, throughput.
- E3. Pareto plots: energy-vs-deadline-miss; energy-vs-network-quality;
  energy-vs-battery-remaining.
- E4. Ablations: drop each scheduler feature (lookahead, contention-awareness,
  priority weighting) and show it matters.

**Exit**: figure set for the paper.

### Phase P — Paper writeup

- P1. Target MLSys / SenSys / HotMobile.
- P2. Open-source repo.

---

## 8. Repo structure

```
research_dev/
├── MULTI_MODEL_PLAN.md             # this file
├── VGGT_PLAN.md                    # single-model reference
├── PROTOTYPE_PLAN.md               # prior VLM plan (preserved)
├── edge_encoder/                   # existing infrastructure — reused
│   ├── run_video_on_device.py      # perfetto harness
│   └── sample_video_frames.py      # preprocessing
├── traces/                         # NEW — Phase W output
│   ├── aria_kitchen_01.json
│   ├── ego4d_conversation_04.json
│   └── trace_schema.md
├── models/
│   ├── vggt_split/                 # mirror VGGT_PLAN.md layout
│   ├── whisper/
│   │   ├── export_qnn.py
│   │   └── phone_run.sh
│   └── llm_server/
│       └── serve.py
├── cost_model/
│   ├── cost_table.json             # (model, device, concurrent) → (lat, Wh, bytes)
│   ├── measure_solo.py             # Phase V2/A2/L3
│   ├── measure_contention.py       # Phase C1
│   └── validate.py                 # Phase C2 assumption checks
├── scheduler/
│   ├── ilp.py                      # Phase S1
│   ├── greedy.py                   # Phase S2
│   ├── baselines.py                # always_local / always_cloud / round_robin / neurosurgeon
│   └── simulate.py                 # trace replay + placement output
├── energy/
│   ├── phone_harness.py            # reuses edge_encoder
│   ├── server_harness.py           # nvidia-smi integration
│   └── aggregate.py                # total Wh across tiers
└── eval/
    ├── run_pareto.py               # Phase E1/E2
    └── plot_*.py                   # Pareto + ablation figures
```

---

## 9. What's already done, what's genuinely new

Already built (directly reused):

- ExecuTorch + QNN SDK + NDK + ADB environment.
- Per-model export recipe (LLaVA + Qwen2-VL working; pattern generalizes).
- Perfetto-based energy harness with charge_uah at 4 Hz, no root.
- Idle baseline methodology (differencing).
- Full push → run → pull pipeline for phone tests.

Genuinely new:

- Workload trace derivation from Aria/Ego4D.
- Whisper and small-LLM pipelines.
- Contention measurement methodology.
- The scheduler itself.
- Multi-device coordinated measurement (two phones + server simultaneously).
- ILP / greedy baselines + Neurosurgeon reimplementation.

---

## 10. Risks & mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Aria Everyday Activities access requires institutional affiliation we don't have | Medium | Fall back to Ego4D (simple form) or synthesize from any egocentric video. |
| Pairwise-contention approximation fails (§4a) | Medium | Report the failure as a paper result; scheduler uses full cost table for 3-task combos and approximates beyond. |
| Scheduler can't beat always-local on realistic traces because network energy wipes out the gain | Low | Expected outcome for low-quality networks; include network-quality as an axis and show the policy adapts. |
| Phi-3 / Llama-3 unavailable on HuggingFace due to gated access | Low | Use Phi-3-mini (Apache) or Mistral-7B-instruct (Apache). |
| Contention numbers noisy (<5% signal) | Medium | Run more iterations; use longer workload bursts. |
| Timeline creeps — each model takes 2–3× planned | High | No deadline, graceful pace. Budget each phase by *exit criterion*, not time. |

---

## 11. Budget

- Hardware: existing (OP11, OP12, A6000). ~$40 of A100 cloud for the
  "cloud tier" in evaluations (6 hours at rental rates).
- Datasets: free (Aria / Ego4D with access form).
- No hard deadline → no opportunity cost pressure.

---

## 12. Near-term checklist (first two weeks)

1. **Fill out Aria Everyday Activities access form** (today).
2. **Fill out Ego4D access form in parallel** — takes 24-48 hours.
3. **Install `projectaria_tools`** and download one short Aria recording;
   inspect its structure (eye / IMU / video / audio synchronization).
4. **Write `traces/derive_from_aria.py`** that takes one recording and
   produces a JSON task-arrival trace via the §3c derivation (VAD, intent
   matching, fps trigger).
5. **Commit the first trace** to `research_dev/traces/`. Single trace is
   enough to unblock the scheduler design work.
6. After 1 trace exists: return to VGGT pipeline (phase V, reuses
   VGGT_PLAN.md), now in service of the cost table rather than as a
   standalone paper.

The workload trace is the scarcest resource. Everything else builds on
existing infrastructure; the trace is what lets us test whether the
scheduler actually matters.

---

## 13. Current status (session checkpoint)

State on entry to a fresh session. Start by reading this.

### 13a. Infrastructure (all working)

- ExecuTorch + QNN SDK 2.45 + Android NDK r27c on the Linux host.
- OP12 (SM8650) connected via wireless ADB; OP11 (SM8550) available.
- Perfetto-based energy harness at 4 Hz `batt.charge_uah`, no root.
  Handles unit conversion correctly (uAh × V / 1e6 = Wh). Entry point:
  [`edge_encoder/run_video_on_device.py`](edge_encoder/run_video_on_device.py).
- Idle baseline on OP12 (screen off, WiFi up): **327 mW**.
- A6000 (48 GB VRAM) available for server-tier experiments.

### 13b. Models characterized and on-device

| Model | Device | Config | Latency | Energy/frame | Cosine | Notes |
|---|---|---|---|---|---|---|
| LLaVA CLIP-L ViT | OP12 HTP | fp16 | 143 ms | 1.7 J | 0.990 | Production choice |
| Qwen2-VL ViT | OP12 HTP | 16a8w | 1.91 s | ~10 J | 0.991 | High-accuracy alternative |
| VGGT-1B full | A6000 | fp16 | 304 ms (S=1) | ~70 J | reference | Cloud baseline |
| VGGT aggregator only | A6000 | fp16 | 261 ms (S=1) | ~60 J | — | Split candidate |
| VGGT heads | A6000 | fp16 | 43 ms (S=1) | ~10 J | — | Server-pinned |

VGGT timings scale super-linearly with S (global attention O(S²)): 261 ms
(S=1) → 628 ms/frame (S=16). See
[`vggt_split/profile_vggt_split.py`](vggt_split/profile_vggt_split.py).

### 13c. Phase 6 sanity (done): edge tokens through cloud LLM

Phone-computed LLaVA ViT embeddings, injected into the cloud Llama-2-7B
via a patched `get_image_features`, produce **94% word-level overlap**
with the stock cloud pipeline answer. Script:
[`edge_encoder/llava/phase6_inputs_embeds.py`](edge_encoder/llava/phase6_inputs_embeds.py).

### 13d. Workload traces (5 real + 2 synthetic)

**Real — derived from Aria Gen2 Pilot dataset** via diarization CSV +
intent-keyword matching. Committed in
[`traces/`](traces/):

| Trace | Duration | VGGT | Whisper | LLM | Character |
|---|---|---|---|---|---|
| aria_eat_0 | 324 s | 648 | 175 | 10 | Social dinner |
| aria_walk_0 | ~300 s | 599 | 129 | 9 | Walking + talk |
| aria_cook_0 | ~335 s | 669 | 77 | 0 | Solo cooking (SELF-only) |
| aria_play_0 | ~342 s | 683 | 164 | 6 | Games + social |
| aria_clean_0 | ~331 s | 661 | 22 | 3 | Mostly quiet |

Diarization CSV has ns-precision `start_timestamp_ns, end_timestamp_ns,
speaker (SELF/OTHER), content`. Real transcripts power intent matching.

**Downloaded Aria data** at [`aria_data/`](aria_data/) — 5 sequences ×
(`video_main_rgb` + `diarization` + `depth` + `mps_slam_trajectories` +
`mps_slam_calibration`), ~14 GB on disk. Depth + trajectories are GT for
future VGGT quality evaluation. Not yet used.

**Synthetic fallback**:
[`traces/synthetic_3min.jsonl`](traces/synthetic_3min.jsonl) — 412
events, built by repeating a real 3.4 s VOiCES speech clip with varied
gaps. Kept for unit-test fidelity.

### 13e. Scheduler (skeleton + 4 policies)

`research_dev/scheduler/`:

- [`cost_table.py`](scheduler/cost_table.py) — (model, device) →
  (latency_ms, energy_J, output_bytes). Populated with **measured** rows
  for LLaVA+Qwen2VL on phone, VGGT aggregator/heads on A6000; the rest
  are clearly-marked placeholders.
- [`policies.py`](scheduler/policies.py) — `always_local`, `always_cloud`,
  `round_robin`, `greedy_energy`.
- [`simulate.py`](scheduler/simulate.py) — trace replay, per-device
  latency accumulation, deadline checks. CLI:

  ```bash
  python -m research_dev.scheduler.simulate \
      --trace research_dev/traces/aria_eat_0.jsonl --policy all
  ```

Scheduler runs end-to-end on all 5 real Aria traces. See §13f for results.

### 13f. Critical finding blocking further scheduler work

**With real VGGT cost (aggregator output = 67 MB / frame),
every static policy fails.** Across all 5 traces:

- `always_local` misses ~99% of VGGT deadlines (phone estimate 800 ms >
  deadline 450 ms at 2 fps).
- `always_cloud` misses 78–97% (67 MB × 0.117 kWh/GB mobile =
  **28,200 J per upload**; latency >5 s on wifi too).
- `round_robin` is worse than always_cloud.
- `greedy_energy` matches always_local's energy but can't fix the
  deadline problem either.

Root cause is **architectural**, not scheduling. The 2 fps × 0.45 s
deadline × 67 MB output combination is infeasible regardless of policy.

Three fixes (any one unblocks the scheduler story):

1. **Compress VGGT output** — ship final-layer only (~2.8 MB) or apply
   VQ-KMeans. Simulator would immediately show greedy beating baselines.
2. **Loosen VGGT deadline** to 2–3 s (accepting that 3D reconstruction
   is soft-realtime, not per-frame). 2 fps × 2 s deadline is realistic
   for AR scene building.
3. **Switch to a streaming model** — CUT3R or Spann3R were designed for
   incremental scene build, one frame at a time, with smaller per-frame
   output. This is a model-choice pivot (not a scheduling improvement).

Each is ~1–2 hours of implementation on top of current infrastructure.

### 13g. Paused / not started

- VGGT on phone (VGGT_PLAN Phase 2) — export aggregator to QNN. Still
  placeholder cost in the table.
- Whisper + small LLM on device — both placeholder cost.
- Contention dimension in cost table (Phase C) — currently solo-only.
- Task dependencies (LLM waits for Whisper transcript).
- ILP offline baseline.

### 13h. Repo tree snapshot

```
research_dev/
├── MULTI_MODEL_PLAN.md                  # primary plan (this file)
├── VGGT_PLAN.md                         # VGGT-specific (single-model)
├── PROTOTYPE_PLAN.md                    # original VLM plan (archived)
├── AriaGen2PilotDataset_download_urls.json  # manifest (signed URLs; CDN expires)
├── aria_data/                           # downloaded Aria (~14 GB)
│   ├── eat_0/  walk_0/  cook_0/  play_0/  clean_0/  walk_1/
├── edge_encoder/                        # LLaVA + Qwen2VL (reused for harness)
│   ├── run_video_on_device.py           # perfetto energy harness
│   ├── sample_video_frames.py
│   ├── llava/phase6_inputs_embeds.py
│   └── ...
├── scheduler/
│   ├── cost_table.py                    # (model, device) costs
│   ├── policies.py                      # 4 baseline policies
│   └── simulate.py                      # trace replay
├── traces/
│   ├── derive_from_aria.py              # derivation script (VAD or diarization)
│   ├── aria_download.py                 # manifest-driven downloader
│   ├── aria_{eat,walk,cook,play,clean}_0.jsonl   # 5 real traces
│   ├── synthetic_3min.jsonl             # fallback
│   └── sample_speech.jsonl              # unit-test fixture
└── vggt_split/
    ├── run_vggt_smoke.py                # end-to-end forward
    └── profile_vggt_split.py            # aggregator-alone timing
```

### 13i. Resume checklist

When you come back:

1. Re-source env if needed: `conda activate research`. Check adb
   (`adb devices`) — may need to reconnect wireless ADB.
2. Read this status section and §13f.
3. Pick ONE of the three fixes in §13f to unblock scheduler story.
4. After fix: rerun `python -m research_dev.scheduler.simulate --trace
   research_dev/traces/aria_eat_0.jsonl --policy all` and confirm
   greedy_energy now differentiates from baselines.

The fastest path to a publishable first figure from here is
fix-#2 (loosen deadline) — no code changes, just update the trace's
`deadline_s` field for VGGT events to 2.0. That exposes the scheduler's
value without introducing a new design assumption.
