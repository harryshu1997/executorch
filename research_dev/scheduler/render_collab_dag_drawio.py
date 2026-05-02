"""
Render a draw.io diagram for the 2-phone + (server | cloud) collaboration
pattern. Two stacked panels: top = local server over wifi, bottom = cloud
over cellular. Each panel has lanes for both phones' subsystems
(NPU/CPU/HW264/Radio), plus the server or cloud lane.

Energy per task labelled in each box. Network energy on each upload arrow.
Numbers are a mix of measured and estimated — sources marked.

Synthetic 30 s timeline:
  - VAD-detected utterances at t=4 s and t=18 s (each phone independently)
  - Continuous motion capture on CPU (cheap, always on)
  - Continuous H.264 encode → upload at 5 Mbps avg (one 2 s clip every 2 s)
  - Server/Cloud runs vJEPA2 fpc16 (37 ms, 4.3 J) on each clip
  - Gemma query event at t=10 s
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


# ----- Constants -------------------------------------------------------------

T_MAX = 30.0
PX_PER_S = 38
LABEL_W = 230
ORIGIN_X = LABEL_W + 20
PANEL_W = ORIGIN_X + int(PX_PER_S * T_MAX) + 60
LANE_H = 56
LANE_GAP = 4
HEADER_H = 70
PANEL_GAP = 60

# --- Energy estimates (J, J/s, J/B) ----
# All marginal energy (above phone idle baseline of ~327 mW).
# Phone-side radio energy is DEVICE-ONLY (modem RF + chain), NOT the
# full network-infrastructure-amortized 0.030 kWh/GB / 0.117 kWh/GB
# from the cost_table.py — those are too pessimistic for a single
# phone's battery argument and would dwarf everything else here.

# Measured (this session):
# Whisper-Tiny on OP15 NPU (Hexagon v81 HMX/HVX): 117 ms total
#   (105 ms enc + 11 ms dec + ~1 ms overhead).
# NPU avg power during inference is ~1.5 W (HVX cores 0.2-0.4 W +
# HMX matmul 0.5-1.5 W + DDR reads 0.2-0.5 W). Whisper is mostly
# memory-bound so it sits at ~1.5 W avg, not the 3 W matmul peak.
# Energy ≈ 0.117 s × 1.5 W = 0.176 J. Round to 0.18 J.
ENERGY_WHISPER_PER_UTT = 0.18          # J — 117 ms × 1.5 W avg NPU
ENERGY_VJEPA_SERVER_PER_CLIP = 4.3     # J — measured on A6000 fp16 fpc16
ENERGY_GEMMA_PER_QUERY = 230.0         # J — measured A6000 bf16 50-tok
# Estimated phone-side device-only:
POWER_PHONE_CPU_MOTION = 0.030         # W — light CPU, ~3% util
POWER_PHONE_HW_H264 = 0.030            # W — dedicated HW encoder block
POWER_SERVER_IDLE = 30.0               # W — A6000 idle
POWER_CLOUD_IDLE = 25.0                # W — A100 idle estimate
# Phone modem energy (device-only):
#  WiFi 802.11ac: ~1.5 W tx active, ~9 MB/s real ⇒ 0.167 J/MB
#  4G/5G: ~2.5 W tx active, ~2 MB/s real (LTE)   ⇒ 1.25  J/MB
J_PER_BYTE_WIFI = 1.67e-7              # 0.167 J/MB device-only
J_PER_BYTE_CELL = 1.25e-6              # 1.25  J/MB device-only
# Workload assumptions:
H264_BITRATE_MBPS = 5.0
H264_BYTES_PER_S = H264_BITRATE_MBPS * 1e6 / 8  # 625 KB/s
CLIP_DUR_S = 2.0
CLIP_BYTES = int(H264_BYTES_PER_S * CLIP_DUR_S)  # ~1.25 MB

# Per-clip network energy
WIFI_J_PER_CLIP = CLIP_BYTES * J_PER_BYTE_WIFI         # ~0.135 J
CELL_J_PER_CLIP = CLIP_BYTES * J_PER_BYTE_CELL         # ~0.526 J


def t_to_x(t: float, x_origin: int) -> int:
    return x_origin + int(PX_PER_S * t)


# ----- Cell rendering helpers ------------------------------------------------

CELL = ('<mxCell id="{id}" value="{val}" style="{style}" vertex="1" '
        'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
        'as="geometry"/></mxCell>')
EDGE = ('<mxCell id="{id}" value="{val}" style="{style}" edge="1" parent="1" '
        'source="{src}" target="{dst}"><mxGeometry relative="1" '
        'as="geometry"/></mxCell>')


def box(cid, x, y, w, h, label, fill="#dae8fc", stroke="#6c8ebf",
        font_size=9, rounded=True, font_color="#000"):
    style = (f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};fontSize={font_size};"
             f"fontColor={font_color};align=center;verticalAlign=middle;")
    return CELL.format(id=cid, val=escape(label), style=style,
                       x=x, y=y, w=w, h=h)


def text(cid, x, y, w, h, label, font_size=11, bold=False, align="left",
         color="#000"):
    style = (f"text;html=1;strokeColor=none;fillColor=none;align={align};"
             f"verticalAlign=middle;fontSize={font_size};fontColor={color};"
             f"{'fontStyle=1;' if bold else ''}")
    return CELL.format(id=cid, val=escape(label), style=style,
                       x=x, y=y, w=w, h=h)


def edge(cid, src, dst, label="", color="#666", dashed=False, font_size=8):
    style = (f"endArrow=classic;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
             f"entryX=0.5;entryY=0;strokeColor={color};fontSize={font_size};"
             f"{'dashed=1;' if dashed else ''}")
    return EDGE.format(id=cid, val=escape(label), style=style,
                       src=src, dst=dst)


# ----- Synthetic event timeline (shared between panels) ----------------------

@dataclass
class Task:
    cid: str
    lane: int
    t_start: float
    t_end: float
    label: str
    fill: str
    stroke: str


def gen_phone_lanes(phone_idx: int, link: str) -> list[Task]:
    """Tasks for one phone (NPU / CPU / HW264 / Radio).

    `link` is "wifi" or "cell" — affects radio energy label.
    """
    base_lane = 0 if phone_idx == 1 else 4  # phone 1 lanes 0-3, phone 2 lanes 4-7
    j_per_clip = WIFI_J_PER_CLIP if link == "wifi" else CELL_J_PER_CLIP

    out: list[Task] = []

    # Whisper utterances — phone 1 at t=4, phone 2 at t=18
    utt_t = 4.0 if phone_idx == 1 else 18.0
    out.append(Task(
        cid=f"p{phone_idx}_whisper_{link}",
        lane=base_lane + 0,
        t_start=utt_t, t_end=utt_t + 0.117,
        label=f"Whisper-Tiny\n105 ms enc + 11 ms dec\n@ ~1.5 W NPU avg = {ENERGY_WHISPER_PER_UTT:.2f} J\n(latency measured, power est)",
        fill="#d5e8d4", stroke="#82b366",
    ))

    # Motion capture: continuous green stripe, label energy/sec
    motion_J_total = POWER_PHONE_CPU_MOTION * T_MAX
    out.append(Task(
        cid=f"p{phone_idx}_motion_{link}",
        lane=base_lane + 1,
        t_start=0, t_end=T_MAX,
        label=f"motion / pixel-diff (CPU)  {POWER_PHONE_CPU_MOTION*1000:.0f} mW  → {motion_J_total:.2f} J / 30s  (est)",
        fill="#fff2cc", stroke="#d6b656",
    ))

    # H.264 encode: continuous, encoder is cheap
    h264_J_total = POWER_PHONE_HW_H264 * T_MAX
    out.append(Task(
        cid=f"p{phone_idx}_h264_{link}",
        lane=base_lane + 2,
        t_start=0, t_end=T_MAX,
        label=f"HW H.264 encoder  {POWER_PHONE_HW_H264*1000:.0f} mW  → {h264_J_total:.2f} J / 30s  (est, dedicated block)",
        fill="#f8cecc", stroke="#b85450",
    ))

    # Radio uploads: every 2 s, one 2-s clip = ~1.25 MB → 100 ms wifi or 500 ms cell
    upload_dur = 0.10 if link == "wifi" else 0.50
    n_clips = int(T_MAX / CLIP_DUR_S)
    radio_total = n_clips * j_per_clip
    for i in range(n_clips):
        t = i * CLIP_DUR_S + 0.05  # tiny offset so 1st clip starts at 0.05
        out.append(Task(
            cid=f"p{phone_idx}_up_{link}_{i}",
            lane=base_lane + 3,
            t_start=t, t_end=t + upload_dur,
            label=(f"upload 1.25 MB\n{j_per_clip*1000:.0f} mJ  ({link})"
                   if i == 0 else ""),
            fill="#dae8fc" if link == "wifi" else "#ffe6cc",
            stroke="#6c8ebf" if link == "wifi" else "#d79b00",
        ))
    return out


def gen_compute_lane(link: str, lane: int) -> list[Task]:
    """Server (or cloud) lane: vJEPA2 inference per clip + Gemma query."""
    out: list[Task] = []
    n_clips = int(T_MAX / CLIP_DUR_S)
    # Each phone uploads every 2s; server processes both phones' clips
    # round-robin. Inference is 37 ms each.
    inference_dur = 0.037
    n_inferences = n_clips * 2  # two phones
    for i in range(n_inferences):
        # Stagger inferences right after upload completes
        phone_idx = i % 2 + 1
        clip_idx = i // 2
        t_upload_done = clip_idx * CLIP_DUR_S + 0.05 + (
            0.10 if link == "wifi" else 0.50
        )
        # Inferences are sequential on the server
        t = t_upload_done + (i % 2) * 0.05  # small gap if both arrive at once
        out.append(Task(
            cid=f"compute_vjepa_{link}_{i}",
            lane=lane,
            t_start=t, t_end=t + inference_dur,
            label=(f"vJEPA2 fp16 fpc16\n37 ms · 4.3 J  (measured)"
                   if i == 0 else ""),
            fill="#e1d5e7", stroke="#9673a6",
        ))
    # Gemma query at t=10
    out.append(Task(
        cid=f"compute_gemma_{link}",
        lane=lane,
        t_start=10.0, t_end=11.9,
        label=f"Gemma-4-E2B query\n1.9 s · {ENERGY_GEMMA_PER_QUERY:.0f} J  (measured)",
        fill="#ffe6cc", stroke="#d79b00",
    ))
    return out


# ----- Panel rendering -------------------------------------------------------

PANEL_LANES_LABELS = [
    ("Phone 1 — NPU (Whisper)",    "#d5e8d4", "#82b366"),
    ("Phone 1 — CPU (motion)",     "#fff2cc", "#d6b656"),
    ("Phone 1 — HW H.264 encoder", "#f8cecc", "#b85450"),
    ("Phone 1 — Radio",            "#dae8fc", "#6c8ebf"),
    ("Phone 2 — NPU (Whisper)",    "#d5e8d4", "#82b366"),
    ("Phone 2 — CPU (motion)",     "#fff2cc", "#d6b656"),
    ("Phone 2 — HW H.264 encoder", "#f8cecc", "#b85450"),
    ("Phone 2 — Radio",            "#dae8fc", "#6c8ebf"),
]


def render_panel(prefix: str, y0: int, link: str, compute_label: str,
                 compute_fill: str) -> tuple[list[str], dict]:
    """Render one collaboration panel (server-wifi or cloud-cell).

    Returns the list of cell strings and an energy summary dict.
    """
    cells: list[str] = []
    n_lanes = len(PANEL_LANES_LABELS) + 1  # phones + compute

    # Panel header
    cells.append(text(
        f"{prefix}_title", 20, y0, PANEL_W - 40, 24,
        f"{compute_label}",
        font_size=15, bold=True))

    # Time axis
    axis_y = y0 + 38
    for s in range(0, int(T_MAX) + 1, 2):
        x = t_to_x(s, ORIGIN_X)
        cells.append(box(f"{prefix}_tick_{s}", x, axis_y, 1, 6, "",
                         fill="#000", stroke="#000", rounded=False))
        cells.append(text(f"{prefix}_ticklbl_{s}", x - 12, axis_y + 6, 30, 12,
                          f"{s}s", font_size=8, align="center"))
    cells.append(box(f"{prefix}_axisline", ORIGIN_X, axis_y + 3,
                     int(PX_PER_S * T_MAX), 1, "",
                     fill="#000", stroke="#000", rounded=False))

    lane_y0 = axis_y + 22

    # Phone lanes
    labels = list(PANEL_LANES_LABELS) + [(compute_label,
                                           compute_fill, "#444")]
    for i, (name, fill, stroke) in enumerate(labels):
        y = lane_y0 + i * (LANE_H + LANE_GAP)
        cells.append(box(f"{prefix}_lbl_{i}", 10, y, LABEL_W - 10, LANE_H,
                         name, fill="#fafafa", stroke="#bbb",
                         font_size=10, rounded=False))
        cells.append(box(f"{prefix}_bg_{i}", ORIGIN_X, y,
                         int(PX_PER_S * T_MAX), LANE_H,
                         "", fill=fill, stroke="#dddddd", rounded=False))

    # Tasks for phone 1 + phone 2 + compute
    all_tasks = (gen_phone_lanes(1, link) + gen_phone_lanes(2, link)
                 + gen_compute_lane(link, len(PANEL_LANES_LABELS)))
    for task in all_tasks:
        x = t_to_x(task.t_start, ORIGIN_X)
        w = max(8, t_to_x(task.t_end, ORIGIN_X) - x)
        ly = lane_y0 + task.lane * (LANE_H + LANE_GAP) + 4
        h = LANE_H - 8
        cells.append(box(f"{prefix}_{task.cid}", x, ly, w, h,
                         task.label, fill=task.fill, stroke=task.stroke,
                         font_size=8))

    # Energy summary box at the bottom of the panel
    j_per_clip = WIFI_J_PER_CLIP if link == "wifi" else CELL_J_PER_CLIP
    n_clips_per_phone = int(T_MAX / CLIP_DUR_S)
    radio_per_phone = n_clips_per_phone * j_per_clip
    motion_per_phone = POWER_PHONE_CPU_MOTION * T_MAX
    h264_per_phone = POWER_PHONE_HW_H264 * T_MAX
    whisper_per_phone = ENERGY_WHISPER_PER_UTT  # 1 utterance in 30 s
    total_phone = (radio_per_phone + motion_per_phone
                   + h264_per_phone + whisper_per_phone)
    n_inferences_total = n_clips_per_phone * 2  # 2 phones
    total_compute = (n_inferences_total * ENERGY_VJEPA_SERVER_PER_CLIP
                     + ENERGY_GEMMA_PER_QUERY)
    grand_total = 2 * total_phone + total_compute

    summary_y = lane_y0 + n_lanes * (LANE_H + LANE_GAP) + 14
    summary_lines = [
        f"30 s session energy ({link}):",
        f"  Each phone:  Whisper {whisper_per_phone:.2f} J  +  motion CPU {motion_per_phone:.2f} J  "
        f"+  HW264 {h264_per_phone:.2f} J  +  radio {radio_per_phone:.2f} J  "
        f"=  {total_phone:.2f} J/phone",
        f"  Compute (2 phones × {n_clips_per_phone} clips = {n_inferences_total} inf): "
        f"{n_inferences_total} × 4.3 J + 1 × 230 J Gemma  =  {total_compute:.0f} J",
        f"  TOTAL session: 2 × {total_phone:.1f}  +  {total_compute:.0f}  =  {grand_total:.0f} J",
    ]
    cells.append(box(f"{prefix}_summary",
                     20, summary_y, PANEL_W - 40, 76,
                     "\n".join(summary_lines),
                     fill="#fff", stroke="#444", font_size=10,
                     rounded=True))

    return cells, {
        "phone_total": total_phone,
        "compute_total": total_compute,
        "grand_total": grand_total,
        "radio_per_phone": radio_per_phone,
    }


# ----- Top-level XML build ---------------------------------------------------

def build() -> tuple[str, dict, dict]:
    panel_height = (HEADER_H + 22 + len(PANEL_LANES_LABELS) * (LANE_H + LANE_GAP)
                    + (LANE_H + LANE_GAP) + 90)
    total_h = 60 + panel_height + PANEL_GAP + panel_height + 100

    cells: list[str] = []

    # Top title
    cells.append(text(
        "title", 20, 10, PANEL_W - 40, 28,
        "2-phone collaboration: local server (wifi) vs cloud (cellular) — "
        "30 s synthetic AR/VR session",
        font_size=18, bold=True))
    cells.append(text(
        "subtitle", 20, 40, PANEL_W - 40, 18,
        "Each phone: Whisper on NPU on speech, continuous motion (CPU), "
        "HW H.264 encode, radio uploads 1.25 MB clips every 2 s. "
        "Compute tier runs vJEPA2 fp16 fpc16 on each clip + Gemma on user query.",
        font_size=10, color="#555"))

    panel1_y = 70
    cells1, sum1 = render_panel(
        "wifi", panel1_y, "wifi",
        "Local Server (A6000) ← phones over WiFi", "#e1d5e7")
    cells.extend(cells1)

    panel2_y = panel1_y + panel_height + PANEL_GAP
    cells2, sum2 = render_panel(
        "cell", panel2_y, "cell",
        "Cloud (A100 estimate) ← phones over Cellular", "#ffe6cc")
    cells.extend(cells2)

    # Comparison summary at bottom
    diff_phone = sum2["phone_total"] - sum1["phone_total"]
    diff_grand = sum2["grand_total"] - sum1["grand_total"]
    cmp_y = panel2_y + panel_height + 20
    cmp = [
        "Side-by-side: cloud over cellular costs MORE per phone (radio dominates the diff):",
        f"  per-phone  wifi: {sum1['phone_total']:.2f} J   cell: {sum2['phone_total']:.2f} J   "
        f"Δ = +{diff_phone:.2f} J  (~{100*diff_phone/sum1['phone_total']:.0f}% more per phone)",
        f"  full session  wifi: {sum1['grand_total']:.0f} J   cell: {sum2['grand_total']:.0f} J   "
        f"Δ = +{diff_grand:.0f} J  (~{100*diff_grand/sum1['grand_total']:.0f}% more total)",
        "Note: server vs cloud compute energy assumed equal here. Real cloud may be 10-30% lower per inf "
        "if A100/H100 fp16 is faster than A6000 fp16.",
    ]
    cells.append(box("cmp", 20, cmp_y, PANEL_W - 40, 88,
                     "\n".join(cmp),
                     fill="#f0f0ff", stroke="#446", font_size=10))

    inner = "\n        ".join(cells)
    xml = f"""<mxfile host="research_dev" type="device">
  <diagram name="multi_model_collab" id="collab1">
    <mxGraphModel dx="{PANEL_W}" dy="{total_h}" grid="1" gridSize="10"
                   guides="1" tooltips="1" connect="1" arrows="1" fold="1"
                   page="1" pageScale="1" pageWidth="{PANEL_W}"
                   pageHeight="{total_h}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {inner}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    return xml, sum1, sum2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="research_dev/multi_model_collab.drawio")
    args = ap.parse_args()

    xml, sum_wifi, sum_cell = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"\n30s synthetic session energy (2 phones):")
    print(f"  WiFi+Server:    {sum_wifi['grand_total']:.1f} J  "
          f"(per-phone {sum_wifi['phone_total']:.2f} J · compute {sum_wifi['compute_total']:.0f} J)")
    print(f"  Cellular+Cloud: {sum_cell['grand_total']:.1f} J  "
          f"(per-phone {sum_cell['phone_total']:.2f} J · compute {sum_cell['compute_total']:.0f} J)")
    print(f"  Δ phone radio energy:  +{sum_cell['radio_per_phone'] - sum_wifi['radio_per_phone']:.2f} J/phone "
          f"(cellular = {sum_cell['radio_per_phone']:.2f} J vs wifi {sum_wifi['radio_per_phone']:.2f} J)")


if __name__ == "__main__":
    main()
