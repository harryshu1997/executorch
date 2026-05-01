"""
Replay a task-arrival trace against a scheduling policy. Emits per-task
placements and aggregate metrics (total Wh, deadline miss rate, throughput).

Mapping from trace `task` strings to cost-table model names lives in
TASK_TO_MODEL; adjust when we add models.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .cost_table import CostTable, network_cost
from .policies import PolicyContext, POLICIES, _link_to

TASK_TO_MODEL = {
    "VGGT_encoder": "VGGT_encoder",
    "Whisper":      "Whisper",
    "LLM_query":    "LLM_query",
    # Lets us reuse the LLaVA cost row as a stand-in for VGGT encoder
    # when we want to simulate with measured numbers (set by --alias-vggt).
}


@dataclass
class Placement:
    t_arrival: float
    task: str
    device: str
    latency_s: float
    energy_J: float
    deadline_s: float
    met_deadline: bool


@dataclass
class SimResult:
    placements: list[Placement] = field(default_factory=list)
    policy: str = ""
    total_energy_J: float = 0.0
    deadline_misses: int = 0
    per_device_energy_J: dict[str, float] = field(default_factory=dict)
    per_task_count: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"policy: {self.policy}",
            f"  total_energy: {self.total_energy_J:.1f} J  ({self.total_energy_J/3600*1000:.2f} mWh)",
            f"  deadline_misses: {self.deadline_misses}/{len(self.placements)} "
            f"({100*self.deadline_misses/max(1,len(self.placements)):.1f}%)",
            f"  per_device_energy_J:",
        ]
        for d, e in sorted(self.per_device_energy_J.items()):
            lines.append(f"    {d}: {e:.1f} J")
        lines.append("  per_task_count:")
        for k, v in sorted(self.per_task_count.items()):
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


def simulate(trace_path: Path, policy_name: str, costs: CostTable,
             alias_vggt_to_llava: bool = False) -> SimResult:
    policy = POLICIES[policy_name]
    device_free_at: dict[str, float] = {"phone": 0.0, "server": 0.0, "cloud": 0.0}
    last_device_for_task: dict[str, str] = {}
    result = SimResult(policy=policy_name)

    events = [json.loads(l) for l in open(trace_path)]
    events.sort(key=lambda e: e["t"])

    for ev in events:
        trace_task = ev["task"]
        model = TASK_TO_MODEL.get(trace_task, trace_task)
        if alias_vggt_to_llava and model == "VGGT_encoder":
            model = "LLaVA_vit"
        deadline = float(ev["deadline_s"])
        t_arr = float(ev["t"])

        ctx = PolicyContext(
            now_s=t_arr,
            device_free_at=device_free_at,
            last_device_for_task=last_device_for_task,
        )

        # Policies that take deadline as extra arg (greedy_energy)
        try:
            device = policy(model, ctx, costs, deadline_s=deadline)
        except TypeError:
            device = policy(model, ctx, costs)
        if device is None:
            print(f"WARN: no feasible device for {model} at t={t_arr}")
            continue
        last_device_for_task[model] = device

        c = costs.get(model, device)
        if c is None:
            print(f"WARN: no cost entry for ({model}, {device})")
            continue
        net_s, net_J = network_cost(c.output_bytes, _link_to(device))
        busy_wait = max(0.0, device_free_at[device] - t_arr)
        lat_s = busy_wait + c.latency_ms / 1000.0 + net_s
        e_J = c.energy_J + net_J
        device_free_at[device] = t_arr + busy_wait + c.latency_ms / 1000.0

        met = lat_s <= deadline
        result.placements.append(Placement(
            t_arrival=t_arr, task=model, device=device,
            latency_s=lat_s, energy_J=e_J, deadline_s=deadline, met_deadline=met,
        ))
        result.total_energy_J += e_J
        result.per_device_energy_J[device] = result.per_device_energy_J.get(device, 0) + e_J
        result.per_task_count[model] = result.per_task_count.get(model, 0) + 1
        if not met:
            result.deadline_misses += 1
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="JSONL trace file.")
    ap.add_argument("--policy", default="all",
                    help=f"Policy name, or 'all' to compare. Options: {','.join(POLICIES)}")
    ap.add_argument("--alias_vggt_to_llava", action="store_true",
                    help="Substitute measured LLaVA ViT cost for VGGT_encoder events "
                         "(uses real phone numbers for simulation).")
    args = ap.parse_args()

    costs = CostTable()
    policies_to_run = list(POLICIES) if args.policy == "all" else [args.policy]

    for p in policies_to_run:
        res = simulate(Path(args.trace), p, costs,
                       alias_vggt_to_llava=args.alias_vggt_to_llava)
        print(res.summary())
        print()


if __name__ == "__main__":
    main()
