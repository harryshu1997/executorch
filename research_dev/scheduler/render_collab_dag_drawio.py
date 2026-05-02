"""
Render a draw.io diagram for the 2-phone + (server | cloud) collaboration
pattern. Two stacked panels: top = local server over wifi, bottom = cloud
over cellular. Each panel shows ONE representative phone (both phones are
identical and run in parallel — totals account for ×2). Lane structure:
NPU / CPU / HW264 / Radio / Compute.

Energy per task labelled inside each box. Bigger fonts. 30 s synthetic
timeline, with a clear summary panel for each setup and a comparison block.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


# ----- Layout constants ------------------------------------------------------

T_MAX = 30.0
PX_PER_S = 56                    # time-axis scale (was 38)
LABEL_W = 280                    # left lane-label column
ORIGIN_X = LABEL_W + 30
PANEL_W = ORIGIN_X + int(PX_PER_S * T_MAX) + 60
LANE_H = 80                      # taller lanes for bigger fonts
LANE_GAP = 8
HEADER_H = 110
SUMMARY_H = 130
PANEL_GAP = 50

# --- Energy estimates --------------------------------------------------------
# Whisper-Tiny on OP15 NPU (Hexagon v81): 117 ms × ~1.5 W avg = 0.18 J.
ENERGY_WHISPER_PER_UTT = 0.18
ENERGY_VJEPA_SERVER_PER_CLIP = 4.3     # MEASURED on A6000 fp16 fpc16
ENERGY_GEMMA_PER_QUERY = 230.0         # MEASURED on A6000 bf16 50-tok
POWER_PHONE_CPU_MOTION = 0.030         # estimated
POWER_PHONE_HW_H264 = 0.030            # estimated, dedicated HW block
J_PER_BYTE_WIFI = 1.67e-7              # device-only modem ~0.167 J/MB
J_PER_BYTE_CELL = 1.25e-6              # device-only modem ~1.25 J/MB
H264_BITRATE_MBPS = 5.0
CLIP_DUR_S = 2.0
CLIP_BYTES = int(H264_BITRATE_MBPS * 1e6 / 8 * CLIP_DUR_S)  # ~1.25 MB
WIFI_J_PER_CLIP = CLIP_BYTES * J_PER_BYTE_WIFI    # ~0.21 J / clip
CELL_J_PER_CLIP = CLIP_BYTES * J_PER_BYTE_CELL    # ~1.56 J / clip


def t_to_x(t: float) -> int:
    return ORIGIN_X + int(PX_PER_S * t)


# ----- Cell helpers ----------------------------------------------------------

CELL = ('<mxCell id="{id}" value="{val}" style="{style}" vertex="1" '
        'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
        'as="geometry"/></mxCell>')


def box(cid, x, y, w, h, label, fill="#dae8fc", stroke="#6c8ebf",
        font_size=12, rounded=True, color="#000", bold=False):
    style = (f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};fontSize={font_size};"
             f"fontColor={color};align=center;verticalAlign=middle;"
             f"{'fontStyle=1;' if bold else ''}")
    return CELL.format(id=cid, val=escape(label), style=style,
                       x=x, y=y, w=w, h=h)


def text(cid, x, y, w, h, label, font_size=14, bold=False, align="left",
         color="#000"):
    style = (f"text;html=1;strokeColor=none;fillColor=none;align={align};"
             f"verticalAlign=middle;fontSize={font_size};fontColor={color};"
             f"{'fontStyle=1;' if bold else ''}")
    return CELL.format(id=cid, val=escape(label), style=style,
                       x=x, y=y, w=w, h=h)


# ----- Lane structure --------------------------------------------------------

LANES_PER_PANEL = [
    ("Phone NPU\nWhisper-Tiny",       "#d5e8d4", "#82b366"),
    ("Phone CPU\nmotion / pixel-diff", "#fff2cc", "#d6b656"),
    ("Phone HW H.264\nencoder",        "#f8cecc", "#b85450"),
    ("Phone Radio\n(modem)",           "#dae8fc", "#6c8ebf"),
    ("Compute tier",                    "#e1d5e7", "#9673a6"),
]


@dataclass
class Task:
    cid: str
    lane: int
    t_start: float
    t_end: float
    label: str
    fill: str
    stroke: str
    font_size: int = 11


# ----- Synthesize tasks for one panel ----------------------------------------

def gen_tasks(prefix: str, link: str) -> list[Task]:
    j_per_clip = WIFI_J_PER_CLIP if link == "wifi" else CELL_J_PER_CLIP
    upload_dur = 0.10 if link == "wifi" else 0.50
    out: list[Task] = []

    # Whisper at t=8 (one utterance per phone in a 30s window, single shown)
    out.append(Task(
        cid=f"{prefix}_whisper",
        lane=0, t_start=8.0, t_end=8.117,
        label=f"Whisper-Tiny\n105ms enc + 11ms dec\n0.18 J  (1.5W avg)",
        fill="#d5e8d4", stroke="#82b366", font_size=11,
    ))

    # Motion CPU continuous bar
    motion_J = POWER_PHONE_CPU_MOTION * T_MAX
    out.append(Task(
        cid=f"{prefix}_motion",
        lane=1, t_start=0, t_end=T_MAX,
        label=(f"continuous motion + pixel-diff\n"
               f"30 mW × 30 s  =  {motion_J:.2f} J  (estimate)"),
        fill="#fff2cc", stroke="#d6b656",
    ))

    # H.264 continuous bar
    h264_J = POWER_PHONE_HW_H264 * T_MAX
    out.append(Task(
        cid=f"{prefix}_h264",
        lane=2, t_start=0, t_end=T_MAX,
        label=(f"HW H.264 encoder (continuous)\n"
               f"30 mW × 30 s  =  {h264_J:.2f} J  (estimate)"),
        fill="#f8cecc", stroke="#b85450",
    ))

    # Radio uploads — one box per clip
    n_clips = int(T_MAX / CLIP_DUR_S)
    radio_total = n_clips * j_per_clip
    for i in range(n_clips):
        t = i * CLIP_DUR_S + 0.05
        out.append(Task(
            cid=f"{prefix}_up_{i}",
            lane=3, t_start=t, t_end=t + upload_dur,
            label=(f"upload\n1.25 MB H.264\n{j_per_clip*1000:.0f} mJ"
                   if i == 0 else f"{j_per_clip*1000:.0f} mJ"),
            fill="#dae8fc" if link == "wifi" else "#ffe6cc",
            stroke="#6c8ebf" if link == "wifi" else "#d79b00",
            font_size=10,
        ))

    # Compute: vJEPA2 inferences per clip × 2 phones, plus 1 Gemma query
    inference_dur = 0.037
    gap = 0.05
    for clip_idx in range(n_clips):
        for phone_idx in range(2):
            t_upload_done = clip_idx * CLIP_DUR_S + 0.05 + upload_dur
            t = t_upload_done + phone_idx * gap
            out.append(Task(
                cid=f"{prefix}_vjepa_{clip_idx}_{phone_idx}",
                lane=4, t_start=t, t_end=t + max(0.4, inference_dur),
                # widen visibly so it's not invisible
                label=("vJEPA2 fp16 fpc16\n37 ms × 4.3 J  (measured)"
                       if clip_idx == 0 and phone_idx == 0 else "vJEPA2"),
                fill="#e1d5e7", stroke="#9673a6", font_size=10,
            ))
    out.append(Task(
        cid=f"{prefix}_gemma",
        lane=4, t_start=14.0, t_end=15.9,
        label=f"Gemma-4-E2B query\n1.9 s · 230 J  (measured)",
        fill="#ffe6cc", stroke="#d79b00", font_size=12,
    ))
    return out


# ----- Panel renderer --------------------------------------------------------

def panel_height() -> int:
    return HEADER_H + 30 + len(LANES_PER_PANEL) * (LANE_H + LANE_GAP) + SUMMARY_H


def render_panel(prefix: str, y0: int, link: str, title: str,
                 subtitle: str) -> tuple[list[str], dict]:
    cells: list[str] = []

    # Title block
    cells.append(box(f"{prefix}_title_bg", 20, y0, PANEL_W - 40, 60,
                     "", fill="#e8eef9" if link == "wifi" else "#fff0e0",
                     stroke="#666", rounded=True))
    cells.append(text(f"{prefix}_title", 30, y0 + 4, PANEL_W - 60, 30,
                      title, font_size=20, bold=True))
    cells.append(text(f"{prefix}_sub", 30, y0 + 32, PANEL_W - 60, 24,
                      subtitle, font_size=12, color="#444"))

    axis_y = y0 + 75
    # Time axis ticks every 5 s
    for s in range(0, int(T_MAX) + 1, 5):
        x = t_to_x(s)
        cells.append(box(f"{prefix}_tk_{s}", x, axis_y, 1, 8, "",
                         fill="#000", stroke="#000", rounded=False))
        cells.append(text(f"{prefix}_tklbl_{s}", x - 18, axis_y + 8, 36, 18,
                          f"{s} s", font_size=11, align="center"))
    cells.append(box(f"{prefix}_axis", ORIGIN_X, axis_y + 4,
                     int(PX_PER_S * T_MAX), 1, "",
                     fill="#000", stroke="#000", rounded=False))

    lane_y0 = axis_y + 30

    # Lanes
    for i, (name, fill, stroke) in enumerate(LANES_PER_PANEL):
        y = lane_y0 + i * (LANE_H + LANE_GAP)
        cells.append(box(f"{prefix}_lbl_{i}", 10, y, LABEL_W - 10, LANE_H,
                         name, fill="#fafafa", stroke="#888",
                         font_size=13, bold=True, rounded=False))
        cells.append(box(f"{prefix}_bg_{i}", ORIGIN_X, y,
                         int(PX_PER_S * T_MAX), LANE_H,
                         "", fill=fill, stroke="#bbb", rounded=False))

    # Tasks
    for task in gen_tasks(prefix, link):
        x = t_to_x(task.t_start)
        w = max(12, t_to_x(task.t_end) - x)
        ly = lane_y0 + task.lane * (LANE_H + LANE_GAP) + 6
        h = LANE_H - 12
        cells.append(box(f"{prefix}_{task.cid}", x, ly, w, h, task.label,
                         fill=task.fill, stroke=task.stroke,
                         font_size=task.font_size))

    # Summary panel
    j_per_clip = WIFI_J_PER_CLIP if link == "wifi" else CELL_J_PER_CLIP
    n_clips = int(T_MAX / CLIP_DUR_S)
    radio_per_phone = n_clips * j_per_clip
    motion_per_phone = POWER_PHONE_CPU_MOTION * T_MAX
    h264_per_phone = POWER_PHONE_HW_H264 * T_MAX
    whisper_per_phone = ENERGY_WHISPER_PER_UTT
    total_phone = (radio_per_phone + motion_per_phone
                   + h264_per_phone + whisper_per_phone)
    n_compute = n_clips * 2
    total_compute = n_compute * ENERGY_VJEPA_SERVER_PER_CLIP + ENERGY_GEMMA_PER_QUERY
    grand_total = 2 * total_phone + total_compute

    sy = lane_y0 + len(LANES_PER_PANEL) * (LANE_H + LANE_GAP) + 12
    # Three-column summary: per phone | compute | total
    col_w = (PANEL_W - 60) // 3
    col_x = [20, 20 + col_w + 10, 20 + 2 * (col_w + 10)]

    per_phone_text = (
        f"PER PHONE  (× 2 phones)\n\n"
        f"Whisper × 1 utt:   {whisper_per_phone:.2f} J\n"
        f"motion CPU × 30s:  {motion_per_phone:.2f} J\n"
        f"HW H.264 × 30s:    {h264_per_phone:.2f} J\n"
        f"radio × {n_clips} clips:    {radio_per_phone:.2f} J\n"
        f"────────────────────\n"
        f"Σ per phone =  {total_phone:.2f} J"
    )
    compute_label = "SERVER (A6000)" if link == "wifi" else "CLOUD (A100 est)"
    compute_text = (
        f"{compute_label}\n\n"
        f"vJEPA2 inf × {n_compute}:    {n_compute * ENERGY_VJEPA_SERVER_PER_CLIP:.0f} J\n"
        f"  ({n_compute} × 4.3 J each, fp16 fpc16)\n"
        f"Gemma query × 1:    {ENERGY_GEMMA_PER_QUERY:.0f} J\n"
        f"  (1.9 s, 50-tok response)\n"
        f"────────────────────\n"
        f"Σ compute =  {total_compute:.0f} J"
    )
    total_text = (
        f"30 s SESSION TOTAL\n\n"
        f"2 × {total_phone:.2f} J phones\n"
        f"+ {total_compute:.0f} J compute\n"
        f"────────────────────\n"
        f"GRAND TOTAL =\n  {grand_total:.0f} J"
    )

    cells.append(box(f"{prefix}_sum1", col_x[0], sy, col_w, SUMMARY_H - 16,
                     per_phone_text, fill="#fff", stroke="#82b366",
                     font_size=12, rounded=True))
    cells.append(box(f"{prefix}_sum2", col_x[1], sy, col_w, SUMMARY_H - 16,
                     compute_text, fill="#fff", stroke="#9673a6",
                     font_size=12, rounded=True))
    cells.append(box(f"{prefix}_sum3", col_x[2], sy, col_w, SUMMARY_H - 16,
                     total_text, fill="#fff",
                     stroke="#444",
                     font_size=14, bold=True, rounded=True))

    return cells, {
        "phone_total": total_phone,
        "compute_total": total_compute,
        "grand_total": grand_total,
        "radio_per_phone": radio_per_phone,
    }


# ----- Top-level -------------------------------------------------------------

def build() -> tuple[str, dict, dict]:
    p_h = panel_height()
    cmp_h = 130
    total_h = 60 + p_h + PANEL_GAP + p_h + 30 + cmp_h + 20

    cells: list[str] = []

    cells.append(text(
        "title", 20, 10, PANEL_W - 40, 32,
        "Multi-model collaboration: 2 phones × (Whisper / motion / H.264 / radio) "
        "→ vJEPA2 on server vs cloud",
        font_size=22, bold=True))
    cells.append(text(
        "subtitle", 20, 42, PANEL_W - 40, 22,
        "30 s synthetic AR/VR session  ·  1 utterance per phone  ·  "
        "1 clip every 2 s  ·  1 user query @ t=14 s  ·  energy per step",
        font_size=13, color="#555"))

    panel1_y = 75
    cells1, sum1 = render_panel(
        "wifi", panel1_y, "wifi",
        "Local Server (A6000) over WiFi",
        "Phones upload 1.25 MB H.264 clips every 2 s · radio energy device-only ~0.17 J/MB")
    cells.extend(cells1)

    panel2_y = panel1_y + p_h + PANEL_GAP
    cells2, sum2 = render_panel(
        "cell", panel2_y, "cell",
        "Cloud (A100 est) over Cellular",
        "Same workload, cellular modem device-only ~1.25 J/MB (≈ 7.5× WiFi)")
    cells.extend(cells2)

    # Comparison block
    cy = panel2_y + p_h + 30
    diff_phone = sum2["phone_total"] - sum1["phone_total"]
    diff_grand = sum2["grand_total"] - sum1["grand_total"]
    pct_phone = 100 * diff_phone / sum1["phone_total"]
    pct_grand = 100 * diff_grand / sum1["grand_total"]
    cmp_lines = [
        "SIDE-BY-SIDE  (same compute on each tier; diff is purely radio)",
        "",
        f"  Per-phone marginal energy   wifi  {sum1['phone_total']:>6.2f} J     "
        f"cell  {sum2['phone_total']:>6.2f} J     "
        f"Δ = +{diff_phone:.2f} J  (+{pct_phone:.0f}% per phone)",
        f"  Compute tier total          server {sum1['compute_total']:>5.0f} J     "
        f"cloud {sum2['compute_total']:>5.0f} J     "
        f"Δ = 0 J  (assumed equal; A100 may be 10-30 % less)",
        f"  Grand total (2 phones)      wifi  {sum1['grand_total']:>6.0f} J     "
        f"cell  {sum2['grand_total']:>6.0f} J     "
        f"Δ = +{diff_grand:.0f} J  (+{pct_grand:.0f}% session total)",
        "",
        "Takeaway: at 2-second clip cadence, server compute (≈ 90 % of total) "
        "dominates the radio difference. The phone-side cellular penalty is real "
        "(5×) but small in absolute terms.",
    ]
    cells.append(box("cmp", 20, cy, PANEL_W - 40, cmp_h,
                     "\n".join(cmp_lines), fill="#f5f5ff", stroke="#444",
                     font_size=13))

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
    xml, s1, s2 = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"\n30 s session, 2 phones:")
    print(f"  WiFi+Server:    {s1['grand_total']:>6.1f} J  "
          f"(per-phone {s1['phone_total']:.2f} · compute {s1['compute_total']:.0f})")
    print(f"  Cellular+Cloud: {s2['grand_total']:>6.1f} J  "
          f"(per-phone {s2['phone_total']:.2f} · compute {s2['compute_total']:.0f})")
    print(f"  Δ phone radio: +{s2['radio_per_phone'] - s1['radio_per_phone']:.2f} J/phone "
          f"(cell={s2['radio_per_phone']:.2f}, wifi={s1['radio_per_phone']:.2f})")


if __name__ == "__main__":
    main()
