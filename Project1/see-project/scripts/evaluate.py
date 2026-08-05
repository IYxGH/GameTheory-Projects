#!/usr/bin/env python3
"""Student tool: evaluate your checkpoint against the scripted baselines
before submitting (a private mini-tournament).

    python scripts/evaluate.py submission.pt --episodes 60
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from see.agents.checkpoint import CheckpointAgent               # noqa: E402
from see.tournament.runner import run_tournament                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    args = ap.parse_args()

    agent = CheckpointAgent(args.checkpoint)
    results = run_tournament({f"YOU ({agent.name})": agent},
                             episodes_per_side=args.episodes,
                             seed0=args.seed0, verbose=False)
    print(f"\n{'rank':>4s}  {'agent':30s} {'score':>8s} "
          f"{'as Iran':>8s} {'as U.S.':>8s}")
    for r in results["leaderboard"]:
        marker = "  <-- you" if r["name"].startswith("YOU") else ""
        print(f"{r['rank']:>4d}  {r['name']:30s} {r['score']:8.1f} "
              f"{r['score_as_iran']:8.1f} {r['score_as_us']:8.1f}{marker}")


if __name__ == "__main__":
    main()
