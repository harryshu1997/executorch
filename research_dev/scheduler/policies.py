"""
Scheduling policies. Each picks a device for each task arrival.

A policy is stateless by default; `greedy_energy` inspects current device
load (how busy each device is at the task's arrival time) to decide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .cost_table import CostTable, network_cost


@dataclass
class PolicyContext:
    """Simulator state exposed to a policy at decision time."""
    now_s: float
    device_free_at: dict[str, float]   # earliest time each device is idle
    last_device_for_task: dict[str, str]  # for round-robin tracking


DEVICES = ("phone", "server", "cloud")


def _feasible_devices(task: str, costs: CostTable) -> list[str]:
    return costs.devices_for(task)


def _link_to(device: str) -> str:
    if device == "phone":
        return ""  # stays local
    if device == "server":
        return "phone->server"
    if device == "cloud":
        return "phone->cloud"
    return ""


def always_local(task: str, ctx: PolicyContext, costs: CostTable) -> str:
    """Run on the lowest tier that supports the model."""
    for d in ("phone", "server", "cloud"):
        if d in _feasible_devices(task, costs):
            return d
    raise ValueError(f"no device for {task}")


def always_cloud(task: str, ctx: PolicyContext, costs: CostTable) -> str:
    """Run on the highest tier that supports the model."""
    for d in ("cloud", "server", "phone"):
        if d in _feasible_devices(task, costs):
            return d
    raise ValueError(f"no device for {task}")


def round_robin(task: str, ctx: PolicyContext, costs: CostTable) -> str:
    """Rotate among the feasible devices for this model (ignores cost)."""
    choices = _feasible_devices(task, costs)
    last = ctx.last_device_for_task.get(task)
    if last is None or last not in choices:
        return choices[0]
    idx = (choices.index(last) + 1) % len(choices)
    return choices[idx]


def greedy_energy(task: str, ctx: PolicyContext, costs: CostTable,
                  deadline_s: float | None = None) -> str:
    """Pick the feasible device with min total (compute + network) energy,
    subject to the deadline being met given current device load."""
    best_d, best_J = None, float("inf")
    for d in _feasible_devices(task, costs):
        c = costs.get(task, d)
        if c is None:
            continue
        # latency: wait for device + compute + (if offloading) transfer time
        busy_wait = max(0.0, ctx.device_free_at.get(d, 0.0) - ctx.now_s)
        net_s, net_J = network_cost(c.output_bytes, _link_to(d))
        total_s = busy_wait + c.latency_ms / 1000.0 + net_s
        total_J = c.energy_J + net_J
        if deadline_s is not None and total_s > deadline_s:
            continue
        if total_J < best_J:
            best_J, best_d = total_J, d
    # If no device meets deadline, fall through to cheapest regardless.
    if best_d is None:
        for d in _feasible_devices(task, costs):
            c = costs.get(task, d)
            if c is None: continue
            net_s, net_J = network_cost(c.output_bytes, _link_to(d))
            total_J = c.energy_J + net_J
            if total_J < best_J:
                best_J, best_d = total_J, d
    return best_d  # type: ignore[return-value]


POLICIES: dict[str, Callable] = {
    "always_local": always_local,
    "always_cloud": always_cloud,
    "round_robin":  round_robin,
    "greedy_energy": greedy_energy,
}
