#!/usr/bin/env python3
"""Evaluate ONE submission against the reference pool. JSON to stdout.

This is the sandboxed entrypoint used by the SEEBoard worker (it is also
handy standalone). It is the ONLY place where an uploaded checkpoint's
embedded theory code is executed, so the platform runs it in a separate
resource-limited subprocess.

    python scripts/eval_submission.py submission.pt \
        --episodes 40 --seed0 20260901 --json

Reference pool: the five scripted baselines plus benchmarks/master_agent.pt
(if present). The submission plays every reference in BOTH roles on the
common seed list; the score is the mean raw utility over all episodes.
"""
import argparse
import json
import pathlib
import resource
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_limits(cpu_sec: int = 1800):
    """Defense in depth for untrusted code (container is the real wall).

    Note: no RLIMIT_AS here — CUDA reserves large virtual address ranges
    at init, so an address-space cap breaks GPU evaluation. Real memory
    control belongs to the container (`--memory=8g`).
    """
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    except (ValueError, OSError):
        pass  # not fatal (e.g., inside some containers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-limits", action="store_true")
    args = ap.parse_args()

    if not args.no_limits:
        set_limits()
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    import numpy as np                                    # noqa: E402
    from see.agents.checkpoint import CheckpointAgent     # noqa: E402
    from see.agents.scripted import BASELINES             # noqa: E402
    from see.config import IRAN, US                       # noqa: E402
    from see.env import StrategicEnduranceEnv             # noqa: E402
    from see.tournament.runner import play_episode        # noqa: E402

    sub = CheckpointAgent(args.checkpoint)
    refs = {f"[baseline] {n}": cls() for n, cls in BASELINES.items()}
    master = ROOT / "benchmarks" / "master_agent.pt"
    if master.exists():
        refs["[benchmark] MasterAgent"] = CheckpointAgent(str(master))

    env = StrategicEnduranceEnv()
    seeds = [args.seed0 + k for k in range(args.episodes)]
    per_opponent = {}
    all_i, all_u = [], []
    for name, ref in refs.items():
        u_i = [play_episode(env, {IRAN: sub, US: ref}, s)["u"][IRAN]
               for s in seeds]
        u_u = [play_episode(env, {IRAN: ref, US: sub}, s)["u"][US]
               for s in seeds]
        per_opponent[name] = {
            "as_iran": round(float(np.mean(u_i)), 2),
            "as_us": round(float(np.mean(u_u)), 2),
        }
        all_i += u_i
        all_u += u_u

    result = {
        "team": sub.name,
        "score": round(float(np.mean(all_i + all_u)), 2),
        "score_as_iran": round(float(np.mean(all_i)), 2),
        "score_as_us": round(float(np.mean(all_u)), 2),
        "episodes": len(all_i) + len(all_u),
        "seed0": args.seed0,
        "per_opponent": per_opponent,
        "train_meta": {k: v for k, v in (sub.meta or {}).items()
                       if k in ("total_steps", "iterations")},
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"score={result['score']}  as_I={result['score_as_iran']}  "
              f"as_U={result['score_as_us']}")
        for n, d in per_opponent.items():
            print(f"  vs {n:28s} I:{d['as_iran']:7.1f}  U:{d['as_us']:7.1f}")


if __name__ == "__main__":
    main()
