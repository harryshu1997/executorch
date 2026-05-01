"""
Phase 3 energy harness. Pushes a sampled video (frame_####.raw files +
input_list.txt) to the phone, reads the battery charge counter, runs
qnn_executor_runner looping over all frames, reads the counter again,
and reports Wh + wall time.

Important for a clean measurement:
  - phone on battery (USB charger detached or in data-only mode)
  - screen off; airplane mode
  - device temperature stable (run once to warm silicon, then measure)

Usage:
  python run_video_on_device.py \
    --frames_dir research_dev/edge_encoder/video_frames \
    --pte research_dev/edge_encoder/qnn_16a8w_op12/qwen2vl_vit_qnn_use_16a8w.pte
"""
import argparse
import subprocess
import time
from pathlib import Path

DEV_DIR = "/data/local/tmp/qwen2vl_vit"
CHARGE_PATHS = [
    "/sys/class/power_supply/battery/charge_counter",
    "/sys/class/power_supply/bms/charge_counter",
]
VOLTAGE_PATH = "/sys/class/power_supply/battery/voltage_now"
NOMINAL_BATTERY_VOLTAGE = 3.85  # V; used when exact voltage isn't readable

PERFETTO_CFG = """buffers: { size_kb: 16384 fill_policy: DISCARD }
data_sources: { config {
  name: "android.power"
  android_power_config {
    battery_poll_ms: 250
    battery_counters: BATTERY_COUNTER_CHARGE
    battery_counters: BATTERY_COUNTER_CURRENT
  }
} }
duration_ms: %(dur_ms)d
"""


def start_perfetto(duration_s: float) -> subprocess.Popen:
    """Start a perfetto trace in background covering duration_s seconds."""
    import tempfile
    cfg_local = Path(tempfile.mkstemp(suffix=".pbtxt")[1])
    cfg_local.write_text(PERFETTO_CFG % {"dur_ms": int(duration_s * 1000) + 5000})
    subprocess.run(["adb", "push", str(cfg_local), "/data/local/tmp/cfg.pbtxt"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    adb_shell("rm -f /data/misc/perfetto-traces/run.trace", check=False)
    return subprocess.Popen(
        ["adb", "shell",
         "cat /data/local/tmp/cfg.pbtxt | perfetto --txt -c - "
         "-o /data/misc/perfetto-traces/run.trace"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def parse_perfetto_trace(local_path: str) -> dict:
    """Pull the trace and compute {charge_delta_uah, wall_s, idle_current_est}."""
    subprocess.run(["adb", "pull", "/data/misc/perfetto-traces/run.trace", local_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from perfetto.trace_processor import TraceProcessor
    with TraceProcessor(trace=local_path) as tp:
        rows = list(tp.query("""
            SELECT c.ts, c.value FROM counter c
            JOIN counter_track t ON c.track_id = t.id
            WHERE t.name = 'batt.charge_uah' ORDER BY c.ts
        """))
        if not rows:
            return {}
        ts0, ts1 = rows[0].ts, rows[-1].ts
        return {
            "n_samples": len(rows),
            "dt_s": (ts1 - ts0) / 1e9,
            "charge_start_uah": int(rows[0].value),
            "charge_end_uah": int(rows[-1].value),
            "charge_delta_uah": int(rows[0].value) - int(rows[-1].value),  # positive = discharge
        }


def adb_shell(cmd: str, check: bool = True) -> str:
    return subprocess.run(
        ["adb", "shell", cmd], capture_output=True, text=True, check=check
    ).stdout


def adb_push(src: Path, dst: str) -> None:
    subprocess.run(["adb", "push", "-p", str(src), dst], check=True)


def find_charge_path() -> str | None:
    for p in CHARGE_PATHS:
        if adb_shell(f"cat {p} 2>/dev/null", check=False).strip():
            return p
    return None


def sample_battery(charge_path: str) -> tuple[float, int, int]:
    now = time.time()
    charge_uah = int(adb_shell(f"cat {charge_path}").strip())
    voltage_uv = int(adb_shell(f"cat {VOLTAGE_PATH}").strip())
    return now, charge_uah, voltage_uv


def batterystats_computed_drain_mah() -> float | None:
    """Parse `dumpsys batterystats --charged` for the system computed drain (mAh)."""
    import re
    out = adb_shell("dumpsys batterystats --charged 2>/dev/null", check=False)
    m = re.search(r"Computed drain:\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None


def dumpsys_battery_charge_uah() -> int | None:
    """Returns battery fuel-gauge charge in microamp-hours via `dumpsys battery`.

    Works even when /sys/class/power_supply/battery/charge_counter is not
    readable by shell. Service-level API exposes the same counter.
    """
    import re
    out = adb_shell("dumpsys battery 2>/dev/null", check=False)
    m = re.search(r"Charge counter:\s*([-0-9]+)", out)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="research_dev/edge_encoder/video_frames")
    ap.add_argument(
        "--pte",
        default="research_dev/edge_encoder/qnn_16a8w_op12/qwen2vl_vit_qnn_use_16a8w.pte",
    )
    ap.add_argument("--model_name_on_device", default="qwen2vl_vit_qnn_use_16a8w.pte")
    ap.add_argument("--warm_up", type=int, default=1)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    if not (frames_dir / "input_list.txt").exists():
        raise SystemExit(f"{frames_dir}/input_list.txt not found; run sample_video_frames.py first.")

    n_frames = sum(1 for _ in (frames_dir / "input_list.txt").read_text().strip().splitlines())
    print(f"{n_frames} frames in {frames_dir}")

    # Push model (if host .pte newer) + frames dir.
    print("Pushing model + frames...")
    adb_push(Path(args.pte), f"{DEV_DIR}/{args.model_name_on_device}")
    adb_shell(f"rm -rf {DEV_DIR}/video_frames")
    adb_push(frames_dir, f"{DEV_DIR}/")
    adb_shell(f"mkdir -p {DEV_DIR}/outputs_video")

    cp = find_charge_path()
    print("Resetting batterystats...")
    adb_shell("dumpsys batterystats --reset", check=False)

    if cp:
        b0 = sample_battery(cp)
    drain_before = batterystats_computed_drain_mah()
    dumpsys_uah_before = dumpsys_battery_charge_uah()
    if dumpsys_uah_before is not None:
        print(f"dumpsys battery charge counter before: {dumpsys_uah_before} uAh")

    print(f"Running inference on {n_frames} frames (warm_up={args.warm_up}, iteration=1)...")
    frames_subdir = Path(args.frames_dir).name
    run_cmd = (
        f"cd {DEV_DIR} && "
        f"LD_LIBRARY_PATH=.:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=. "
        f"./qnn_executor_runner "
        f"--model_path {args.model_name_on_device} "
        f"--input_list_path {frames_subdir}/input_list.txt "
        f"--output_folder_path outputs_video "
        f"--warm_up {args.warm_up} "
        f"--iteration 1"
    )
    # Estimate wall time and start a perfetto trace covering the whole run.
    est_per_frame = 2.2 if "qwen" in args.model_name_on_device.lower() else 0.2
    est_wall = est_per_frame * n_frames * (1 + args.warm_up) + 10
    print(f"  starting perfetto trace (estimated wall ~{est_wall:.0f}s)...")
    p_trace = start_perfetto(est_wall)
    time.sleep(2.0)  # let perfetto warm up
    t0 = time.time()
    subprocess.run(["adb", "shell", run_cmd],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    t1 = time.time()
    wall = t1 - t0
    print(f"  inference wall: {wall:.2f} s — waiting for trace to finalize...")
    p_trace.wait()
    perf = parse_perfetto_trace("/tmp/run.trace")
    if perf:
        print(f"  perfetto: {perf['n_samples']} samples over {perf['dt_s']:.1f} s, "
              f"charge {perf['charge_start_uah']} -> {perf['charge_end_uah']} uAh "
              f"(delta {perf['charge_delta_uah']} uAh)")

    print("\n--- LATENCY ---")
    print(f"  wall time:          {wall:.2f} s ({wall/n_frames*1000:.1f} ms/frame incl warmup)")

    if perf and perf.get("charge_delta_uah", 0) > 0:
        duah = perf["charge_delta_uah"]
        trace_dt = perf["dt_s"]
        # uAh -> Ah (/1e6); Ah * V = Wh.
        energy_wh = duah / 1e6 * NOMINAL_BATTERY_VOLTAGE
        avg_ua = duah / (trace_dt / 3600.0)
        print("\n--- ENERGY (perfetto charge_uah, ~4 Hz, covers run + perfetto pad) ---")
        print(f"  charge delta:      {duah/1000:.3f} mAh ({duah} uAh) over {trace_dt:.1f} s")
        print(f"  avg current:       {avg_ua/1000:.1f} mA (implied)")
        print(f"  energy total:      {energy_wh*1000:.2f} mWh  ({energy_wh*3600:.2f} J)")
        print(f"  energy per frame:  {energy_wh*3600*1000/n_frames:.1f} mJ (WARN: includes idle outside inference window)")

    if cp:
        b1 = sample_battery(cp)
        dcharge_uah = b0[1] - b1[1]  # discharge positive
        avg_v = (b0[2] + b1[2]) / 2.0 / 1e6
        energy_wh = dcharge_uah * avg_v / 1e6 / 1000.0
        print("\n--- ENERGY (sysfs charge_counter) ---")
        print(f"  charge delta:      {dcharge_uah/1000:.3f} mAh")
        print(f"  avg voltage:       {avg_v:.3f} V")
        print(f"  energy total:      {energy_wh*1000:.2f} mWh  ({energy_wh*3600:.2f} J)")
        print(f"  energy per frame:  {energy_wh*3600*1000/n_frames:.1f} mJ")

    dumpsys_uah_after = dumpsys_battery_charge_uah()
    if dumpsys_uah_before is not None and dumpsys_uah_after is not None:
        duah = dumpsys_uah_before - dumpsys_uah_after  # discharge positive
        energy_wh = duah / 1e6 * NOMINAL_BATTERY_VOLTAGE
        print("\n--- ENERGY (dumpsys battery 'Charge counter', uAh resolution) ---")
        print(f"  charge delta:      {duah/1000:.3f} mAh ({duah} uAh)")
        print(f"  energy total (~):  {energy_wh*1000:.2f} mWh  ({energy_wh*3600:.2f} J)")
        print(f"  energy per frame:  {energy_wh*3600*1000/n_frames:.1f} mJ")

    drain_after = batterystats_computed_drain_mah()
    if drain_before is not None and drain_after is not None:
        dmah = drain_after - drain_before
        energy_wh = dmah * NOMINAL_BATTERY_VOLTAGE / 1000.0
        print("\n--- ENERGY (dumpsys batterystats 'Computed drain', mAh resolution) ---")
        print(f"  drain delta:       {dmah:.2f} mAh")
        print(f"  energy total (~):  {energy_wh*1000:.2f} mWh  ({energy_wh*3600:.2f} J)")
        print(f"  energy per frame:  {energy_wh*3600*1000/n_frames:.1f} mJ")
    print()
    print("Caveats: system-wide battery delta (screen/radios/bg processes included).")
    print("For short runs (< 1 mAh), numbers are noisy — use 100+ frames for reliable Wh.")


if __name__ == "__main__":
    main()
