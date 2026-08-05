#!/usr/bin/env python3
"""Train your submission. This is the ONLY command you need:

    python scripts/train.py --theory submission_template/theory.py \
        --team "team-name" --steps 300000 --out my_submission.pt

The trainer self-plays the standard network pair (Iran-side + US-side) on
the canonical environment, using YOUR TheorySpec for belief features and
reward shaping. The resulting .pt file is your competition submission.

Tips:
  * ~300k steps takes roughly 15-40 min on a laptop CPU. More helps, with
    diminishing returns. --steps 30000 gives a quick sanity run.
  * Watch the printed u_I / u_U (raw utilities vs the opponent mix). If
    they collapse toward the exit values, your shaping may be teaching the
    agent to give up; if episode length pins at T_bar, it may never exit.
  * Train a few seeds and submit the checkpoint that evaluates best:
        python scripts/evaluate.py my_submission.pt
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from see.training.selfplay import PPOConfig, SelfPlayTrainer   # noqa: E402
from see.training.theory_api import load_theory                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", required=True,
                    help="path to your theory.py")
    ap.add_argument("--team", default="anonymous")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="submission.pt")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--p-self", type=float, default=0.6)
    ap.add_argument("--resume", default=None,
                    help="warm-start nets from a previous checkpoint")
    args = ap.parse_args()

    theory, source = load_theory(args.theory)
    print(f"theory: {theory.name} "
          f"(+{theory.extra_feature_dim} belief features)")
    cfg = PPOConfig(total_steps=args.steps, seed=args.seed, lr=args.lr,
                    entropy_coef=args.entropy, p_self=args.p_self)
    trainer = SelfPlayTrainer(theory, cfg)
    if args.resume:
        trainer.load_nets(args.resume)
    trainer.train()
    trainer.save_checkpoint(args.out, theory_source=source, team=args.team)


if __name__ == "__main__":
    main()
