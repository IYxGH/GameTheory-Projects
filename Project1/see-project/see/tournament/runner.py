"""Tournament runner - the Kaggle-style competition backend.

Protocol (fairness by construction):
  * Every entrant plays every other entrant AND every scripted baseline.
  * Every pairing is played in BOTH role assignments (once as Iran, once as
    the U.S.) over the SAME fixed seed list, so all matchups experience
    identical worlds (identical type draws, endurance shocks, mediation
    luck). Skill differences, not seed luck, drive the ranking.
  * Score = mean raw utility u_i (the paper's payoff - never the shaped
    training reward), reported per role and combined, with bootstrap CIs.
  * Agents receive ONLY (obs, legal_mask, public_state). `info` and the
    true hidden state never reach an agent.

Outputs results.json, consumed by leaderboard.py.
"""
from __future__ import annotations

import itertools
import json
import time
from typing import Dict, List, Optional

import numpy as np

from ..agents.base import Agent
from ..agents.scripted import BASELINES
from ..config import IRAN, PLAYERS, US
from ..env import StrategicEnduranceEnv


def play_episode(env: StrategicEnduranceEnv, agents: Dict[str, Agent],
                 seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    for role, agent in agents.items():
        # per-(episode, role) agent seed, decoupled from the world seed
        agent.reset(role, seed=seed * 7919 + (1 if role == IRAN else 2))
    done = False
    while not done:
        public = env.public_state()
        actions = {}
        for role, agent in agents.items():
            mask = env.get_legal_mask(role)
            try:
                a = agent.act(obs[role], mask, public)
                if not mask[int(a)]:
                    a = int(np.argmax(mask))
            except Exception:
                a = int(np.argmax(mask))       # crashing agent: forced move
            actions[role] = int(a)
        obs, rew, term, trunc, info = env.step(actions)
        done = term[IRAN] or trunc[IRAN]
    truth = info["_true_state"]
    exits = {p: env.exited[p] for p in PLAYERS}
    return {
        "seed": seed,
        "len": env.t,
        "u": {p: truth["utility"][p] for p in PLAYERS},
        "exited": exits,
        "timeout": not any(exits.values()),
    }


def run_tournament(entries: Dict[str, Agent],
                   episodes_per_side: int = 100,
                   seed0: int = 20_260_713,
                   include_baselines: bool = True,
                   verbose: bool = True) -> dict:
    """entries: {name: Agent}. Returns the full results structure."""
    agents = dict(entries)
    baseline_names = []
    if include_baselines:
        for bname, cls in BASELINES.items():
            key = f"[baseline] {bname}"
            agents[key] = cls()
            baseline_names.append(key)

    seeds = [seed0 + k for k in range(episodes_per_side)]
    env = StrategicEnduranceEnv()
    names = sorted(agents)
    pair_results: List[dict] = []
    t0 = time.time()
    pairs = [(a, b) for a, b in itertools.product(names, names) if a != b]
    for k, (name_I, name_U) in enumerate(pairs):
        rows = [play_episode(env, {IRAN: agents[name_I],
                                   US: agents[name_U]}, s) for s in seeds]
        pair_results.append({
            "iran": name_I, "us": name_U,
            "u_I": float(np.mean([r["u"][IRAN] for r in rows])),
            "u_U": float(np.mean([r["u"][US] for r in rows])),
            "len": float(np.mean([r["len"] for r in rows])),
            "iran_outlasts": float(np.mean(
                [r["exited"][US] and not r["exited"][IRAN] for r in rows])),
            "us_outlasts": float(np.mean(
                [r["exited"][IRAN] and not r["exited"][US] for r in rows])),
            "timeout_rate": float(np.mean([r["timeout"] for r in rows])),
            "raw_u_I": [round(r["u"][IRAN], 3) for r in rows],
            "raw_u_U": [round(r["u"][US], 3) for r in rows],
        })
        if verbose:
            pr = pair_results[-1]
            print(f"[{k + 1:3d}/{len(pairs)}] I={name_I:28s} "
                  f"U={name_U:28s} u_I={pr['u_I']:7.1f} "
                  f"u_U={pr['u_U']:7.1f} len={pr['len']:5.1f}", flush=True)

    # aggregate per entrant
    table = []
    rng = np.random.default_rng(0)
    for name in names:
        as_I = [u for pr in pair_results if pr["iran"] == name
                for u in pr["raw_u_I"]]
        as_U = [u for pr in pair_results if pr["us"] == name
                for u in pr["raw_u_U"]]
        combined = np.array(as_I + as_U)
        boots = [float(np.mean(rng.choice(combined, combined.size)))
                 for _ in range(400)]
        table.append({
            "name": name,
            "baseline": name in baseline_names,
            "score": float(combined.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "score_as_iran": float(np.mean(as_I)),
            "score_as_us": float(np.mean(as_U)),
            "episodes": int(combined.size),
        })
    table.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(table, 1):
        row["rank"] = rank
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes_per_side": episodes_per_side,
        "seed0": seed0,
        "elapsed_sec": round(time.time() - t0, 1),
        "leaderboard": table,
        "pairs": pair_results,
    }


def load_entries(submission_dir: str, sample: bool = True,
                 prefer_filename: bool = False) -> Dict[str, Agent]:
    """Load every *.pt checkpoint in a directory as an entrant.

    prefer_filename: name entries by file stem instead of the checkpoint's
    embedded team string — used by the platform's official run so board
    names are the authenticated student IDs, not self-chosen labels."""
    import pathlib

    from ..agents.checkpoint import CheckpointAgent   # lazy: needs torch
    entries: Dict[str, Agent] = {}
    for p in sorted(pathlib.Path(submission_dir).glob("*.pt")):
        try:
            agent = CheckpointAgent(str(p), sample=sample)
            if prefer_filename:
                name = p.stem
            else:
                name = agent.name if agent.name != "anonymous" else p.stem
            base = name
            k = 2
            while name in entries:                    # de-duplicate names
                name = f"{base}-{k}"
                k += 1
            entries[name] = agent
        except Exception as e:                        # noqa: BLE001
            print(f"!! could not load {p.name}: {e}")
    return entries


def save_results(results: dict, path: str = "results.json"):
    slim = dict(results)
    with open(path, "w") as f:
        json.dump(slim, f, indent=1)
    print(f"results -> {path}")
