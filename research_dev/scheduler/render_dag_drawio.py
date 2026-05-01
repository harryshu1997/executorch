"""
Generate a draw.io XML showing the 3-model AR/VR scheduler DAG with a 15 s
synthetic timeline.

Lanes (top → bottom):
  - Triggers     (frame, voice-activity, intent)
  - OP15 NPU     (StreamVGGT aggregator + heads via QNN HMX/HVX)
  - OP15 NPU     (Whisper-Tiny via QNN HVX)
  - Network      (Whisper transcript up + Gemma response down)
  - Server A6000 (Gemma-4-E2B prefill + decode)
  - Outputs      (per-frame depth+pose, transcript, LLM answer)

Synthetic timeline:
  - VGGT @ 2 fps for 15 s → 30 frames, ~200 ms compute each on OP15 NPU.
  - Speech utterance from t=4.0–6.0 s; Silero-VAD endpoints at t=6.0 s.
  - Whisper-Tiny on OP15 NPU starts at t=6.0 s, finishes at t=6.5 s.
  - Intent classifier (cheap, on-CPU) detects question → Gemma trigger.
  - Transcript (~50 KB) uploaded over wifi (~5 ms; visualised wider for clarity).
  - Gemma-4-E2B prefill (200 ms) + decode 50 tokens @ 30 tok/s (1.67 s)
    on A6000, response (~250 KB) downlinked.
  - LLM answer surfaces at t≈8.3 s.

Numbers are illustrative (matching cost_table.py with HMX/HVX optimistic phone
estimates of ~200 ms StreamVGGT, ~500 ms Whisper). Intended for paper figures
and design discussions, not as ground truth — replace with measured numbers
once Phase V/A/L close.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


# ----- Layout constants -------------------------------------------------------

PX_PER_S = 60          # 1 s of timeline = 60 px
T_MAX = 15.0           # seconds
LABEL_W = 200          # width of left-side lane labels
ORIGIN_X = LABEL_W + 20  # x where t=0 begins
WIDTH = ORIGIN_X + int(PX_PER_S * T_MAX) + 60

LANE_H = 70
LANES = [
    ("Triggers",        "#fff2cc", "#d6b656"),  # yellow
    ("OP15 NPU — StreamVGGT", "#dae8fc", "#6c8ebf"),  # blue
    ("OP15 NPU — Whisper-Tiny", "#d5e8d4", "#82b366"),  # green
    ("Network (wifi)",  "#f5f5f5", "#666666"),  # gray
    ("Server A6000 — Gemma-4-E2B", "#ffe6cc", "#d79b00"),  # orange
    ("Outputs",         "#e1d5e7", "#9673a6"),  # purple
]
HEADER_H = 90
LANE_GAP = 6
LANE_Y0 = HEADER_H + 30  # y where lane area begins (after time axis)
HEIGHT = LANE_Y0 + (LANE_H + LANE_GAP) * len(LANES) + 40


def t_to_x(t: float) -> int:
    return ORIGIN_X + int(PX_PER_S * t)


def lane_y(i: int) -> int:
    return LANE_Y0 + i * (LANE_H + LANE_GAP)


# ----- Cells ------------------------------------------------------------------

CELL = '<mxCell id="{id}" value="{val}" style="{style}" vertex="1" parent="1">' \
       '<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'

EDGE = '<mxCell id="{id}" value="{val}" style="{style}" edge="1" parent="1" ' \
       'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>'


def box(cid: str, x: int, y: int, w: int, h: int, label: str,
        fill: str = "#dae8fc", stroke: str = "#6c8ebf",
        font_size: int = 10, rounded: bool = True,
        font_color: str = "#000000") -> str:
    style = (
        f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fontSize={font_size};"
        f"fontColor={font_color};align=center;verticalAlign=middle;"
    )
    return CELL.format(id=cid, val=escape(label), style=style,
                       x=x, y=y, w=w, h=h)


def text(cid: str, x: int, y: int, w: int, h: int, label: str,
         font_size: int = 11, bold: bool = False, align: str = "left",
         font_color: str = "#000000") -> str:
    style = (
        f"text;html=1;strokeColor=none;fillColor=none;align={align};"
        f"verticalAlign=middle;fontSize={font_size};"
        f"fontColor={font_color};"
        f"{'fontStyle=1;' if bold else ''}"
    )
    return CELL.format(id=cid, val=escape(label), style=style,
                       x=x, y=y, w=w, h=h)


def edge(cid: str, src: str, dst: str, label: str = "",
         color: str = "#000000", dashed: bool = False,
         font_size: int = 9, end_arrow: str = "classic") -> str:
    style = (
        f"endArrow={end_arrow};html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
        f"entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor={color};"
        f"fontSize={font_size};"
        f"{'dashed=1;' if dashed else ''}"
    )
    return EDGE.format(id=cid, val=escape(label), style=style, src=src, dst=dst)


# ----- Synthetic data (must match the docstring) ------------------------------

@dataclass
class TaskBox:
    cid: str
    lane: int
    t_start: float
    t_end: float
    label: str
    fill: str
    stroke: str


def synth_tasks() -> list[TaskBox]:
    out: list[TaskBox] = []

    # VGGT @ 2 fps; 200 ms compute each on OP15 NPU (HMX/HVX optimistic).
    vggt_dur = 0.20
    for i in range(int(T_MAX * 2)):
        t = i * 0.5
        out.append(TaskBox(
            cid=f"vggt_{i}",
            lane=1,
            t_start=t,
            t_end=t + vggt_dur,
            label=f"VGGT f{i}\n200 ms" if i < 4 or i % 5 == 0 else "",
            fill="#dae8fc", stroke="#6c8ebf",
        ))

    # Whisper-Tiny: 500 ms on OP15 NPU after VAD endpoint at t=6.0 s.
    out.append(TaskBox(
        cid="whisper",
        lane=2,
        t_start=6.0, t_end=6.5,
        label="Whisper-Tiny\n39M params, 500 ms\n(audio 4–6s)",
        fill="#d5e8d4", stroke="#82b366",
    ))

    # Network: transcript up at t=6.5; response down starting t=8.3.
    out.append(TaskBox(
        cid="net_up",
        lane=3,
        t_start=6.50, t_end=6.55,
        label="↑ transcript 50 KB",
        fill="#eeeeee", stroke="#666666",
    ))
    out.append(TaskBox(
        cid="net_down",
        lane=3,
        t_start=8.30, t_end=8.45,
        label="↓ answer 250 KB",
        fill="#eeeeee", stroke="#666666",
    ))

    # Gemma-4-E2B on A6000: 200 ms prefill + 50 tokens × 33 ms decode = 1.85 s.
    out.append(TaskBox(
        cid="gemma_prefill",
        lane=4,
        t_start=6.55, t_end=6.75,
        label="Gemma-4-E2B prefill\n200 ms",
        fill="#ffe6cc", stroke="#d79b00",
    ))
    out.append(TaskBox(
        cid="gemma_decode",
        lane=4,
        t_start=6.75, t_end=8.30,
        label="Gemma-4-E2B decode\n50 tok @ 30 tok/s",
        fill="#ffe6cc", stroke="#d79b00",
    ))

    return out


def synth_outputs() -> list[TaskBox]:
    """Output markers in the bottom 'Outputs' lane."""
    out: list[TaskBox] = []
    # VGGT head outputs every 0.5 s — show first few to avoid clutter.
    for i in range(0, int(T_MAX * 2), 5):
        t_done = i * 0.5 + 0.20
        out.append(TaskBox(
            cid=f"out_vggt_{i}",
            lane=5,
            t_start=t_done, t_end=t_done + 0.18,
            label=f"depth+pose\n6.4 MB",
            fill="#e1d5e7", stroke="#9673a6",
        ))
    out.append(TaskBox(
        cid="out_whisper", lane=5,
        t_start=6.5, t_end=6.7,
        label="transcript\n50 KB",
        fill="#e1d5e7", stroke="#9673a6",
    ))
    out.append(TaskBox(
        cid="out_gemma", lane=5,
        t_start=8.30, t_end=8.65,
        label="LLM answer\nspeech-out",
        fill="#e1d5e7", stroke="#9673a6",
    ))
    return out


def synth_triggers() -> list[tuple[str, float, str]]:
    """(cell_id, t, label) — trigger markers in lane 0."""
    out: list[tuple[str, float, str]] = []
    for i in range(int(T_MAX * 2)):
        t = i * 0.5
        out.append((f"trig_v_{i}", t, "▼"))  # frame trigger
    out.append(("trig_speech_start", 4.0, "🎤 speech start"))
    out.append(("trig_speech_end",   6.0, "🎤 VAD endpoint"))
    out.append(("trig_intent",       6.5, "🤖 intent → LLM"))
    return out


# ----- XML build --------------------------------------------------------------

def build() -> str:
    cells: list[str] = []

    # Header
    cells.append(text("hdr_title", 20, 10, WIDTH - 40, 28,
                      "Multi-model AR/VR scheduler — 15 s synthetic timeline",
                      font_size=18, bold=True))
    cells.append(text("hdr_sub", 20, 40, WIDTH - 40, 22,
                      "VGGT-stream (vision) + Whisper-Tiny (speech) + Gemma-4-E2B-it (LLM) "
                      "on OP15 NPU (Hexagon v81: HMX prefill / HVX decode) + A6000 server tier.",
                      font_size=11))
    cells.append(text("hdr_note", 20, 62, WIDTH - 40, 22,
                      "Phone numbers are HMX-optimistic estimates; replace with measured values "
                      "once StreamVGGT/Whisper QNN exports land (Phase V2/A2).",
                      font_size=10, font_color="#666666"))

    # Time axis
    axis_y = HEADER_H
    for s in range(0, int(T_MAX) + 1):
        x = t_to_x(s)
        cells.append(box(
            f"tick_{s}", x, axis_y, 1, 8,
            "", fill="#000000", stroke="#000000", rounded=False,
        ))
        cells.append(text(f"tick_lbl_{s}", x - 12, axis_y + 8, 30, 14,
                          f"{s} s", font_size=9, align="center"))
    # axis line
    cells.append(box("axis_line", ORIGIN_X, axis_y + 4, int(PX_PER_S * T_MAX), 1,
                     "", fill="#000000", stroke="#000000", rounded=False))

    # Lanes
    for i, (name, fill, stroke) in enumerate(LANES):
        y = lane_y(i)
        # Lane label (left)
        cells.append(box(f"lane_lbl_{i}", 10, y, LABEL_W - 10, LANE_H,
                         name, fill="#fafafa", stroke="#cccccc",
                         font_size=11, rounded=False))
        # Lane background
        cells.append(box(f"lane_bg_{i}", ORIGIN_X, y, int(PX_PER_S * T_MAX), LANE_H,
                         "", fill=fill, stroke="#dddddd", rounded=False))

    # Trigger markers (lane 0)
    for cid, t, lbl in synth_triggers():
        x = t_to_x(t)
        if lbl == "▼":
            # Small vertical tick.
            cells.append(box(cid, x - 3, lane_y(0) + 8, 6, 12, "▼",
                             fill="#cccccc", stroke="#999999", font_size=10))
        else:
            cells.append(box(cid, x - 60, lane_y(0) + 35, 120, 22, lbl,
                             fill="#fff2cc", stroke="#d6b656", font_size=9))

    # Task boxes
    for tb in synth_tasks() + synth_outputs():
        x = t_to_x(tb.t_start)
        w = max(8, t_to_x(tb.t_end) - x)
        y = lane_y(tb.lane) + 6
        h = LANE_H - 12
        cells.append(box(tb.cid, x, y, w, h, tb.label,
                         fill=tb.fill, stroke=tb.stroke, font_size=9))

    # Dependency edges (informative, sparse).
    cells.append(edge("e1", "trig_speech_end", "whisper",
                      "VAD trigger", color="#82b366"))
    cells.append(edge("e2", "whisper", "net_up",
                      "transcript", color="#666666"))
    cells.append(edge("e3", "net_up", "gemma_prefill",
                      "phone→server", color="#666666"))
    cells.append(edge("e4", "gemma_decode", "net_down",
                      "stream tokens", color="#666666"))
    cells.append(edge("e5", "net_down", "out_gemma",
                      "answer", color="#9673a6"))
    # VGGT context edge into Gemma prefill (optional fusion).
    cells.append(edge("e6", "vggt_12", "gemma_prefill",
                      "scene ctx (opt.)", color="#6c8ebf", dashed=True))

    # Legend
    legend_y = HEIGHT - 30
    cells.append(text("legend_lbl", 10, legend_y, 80, 20, "Legend:",
                      font_size=10, bold=True))
    spec = [
        ("VGGT (HMX matmul)", "#dae8fc", "#6c8ebf"),
        ("Whisper (HVX)",     "#d5e8d4", "#82b366"),
        ("Gemma server",      "#ffe6cc", "#d79b00"),
        ("Network",           "#eeeeee", "#666666"),
        ("Output",            "#e1d5e7", "#9673a6"),
    ]
    lx = 90
    for i, (name, fill, stroke) in enumerate(spec):
        cells.append(box(f"leg_{i}", lx + i * 160, legend_y, 14, 14, "",
                         fill=fill, stroke=stroke, rounded=True))
        cells.append(text(f"leg_lbl_{i}", lx + i * 160 + 18, legend_y, 140, 14,
                          name, font_size=10))

    inner = "\n        ".join(cells)
    return f"""<mxfile host="research_dev" type="device">
  <diagram name="multi_model_dag" id="dag1">
    <mxGraphModel dx="{WIDTH}" dy="{HEIGHT}" grid="1" gridSize="10" guides="1"
                   tooltips="1" connect="1" arrows="1" fold="1" page="1"
                   pageScale="1" pageWidth="{WIDTH}" pageHeight="{HEIGHT}"
                   math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {inner}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/multi_model_dag.drawio")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"open with: drawio --no-sandbox {out} (or upload to https://app.diagrams.net)")


if __name__ == "__main__":
    main()
