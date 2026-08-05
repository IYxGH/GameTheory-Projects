# SEE — Strategic Endurance Engine

Final-project stack for the Game Theory course, operationalizing
**"Strategic Endurance under Uncertainty" (2026)**: a multi-stage war of
attrition with incomplete information, costly signaling, evolving
endurance, and history-dependent movers — as a multi-agent RL environment,
a turnkey self-play trainer, and a Kaggle-style class tournament.

```
students:  theory.py (beliefs + shaping)  ──►  train.py  ──►  submission.pt
                     ▲                                             │
        §4 theoretical analysis (manual)                           ▼
instructor:  submissions/*.pt  ──►  run_tournament.py  ──►  leaderboard.html
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                    # sanity: environment intact
cp -r submission_template my_team             # your workspace
# 1. edit my_team/theory.py  (this is the project)
python scripts/train.py --theory my_team/theory.py --team "my-team" \
       --steps 300000 --out submission.pt     # ~15–40 min on a laptop CPU
python scripts/evaluate.py submission.pt      # vs the 5 reference agents
```

For more information read:

`docs/DESIGN.md` (every formula and parameter of the world)

`see/env.py` docstrings (observation/action layout).
