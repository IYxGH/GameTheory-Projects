# Your submission workspace

1. **Edit `theory.py`** — the only file you write. It has two hooks; WHAT
   goes in each is the content of your Section 4 analysis, not something
   this template gives you.
   - `extra_features(...)`: return a fixed-length vector of numbers, each
     computed from your own observation and the public history, that you
     want the policy to see (Section 4.3). Set `extra_feature_dim` to the
     length you return.
   - `shaping(...)`: an optional training-time term added to the reward
     (Sections 4.1, 4.4, 4.5). The leaderboard scores the game's RAW
     utility, so shaping only changes what your agent learns to want —
     non-telescoping bonuses backfire; potential-based terms are safe.

   The template ships as a null theory (no features, no shaping) so it
   trains out of the box; replace the two methods with your own.

2. **Train** (from the repo root):
   ```bash
   python scripts/train.py --theory submission_template/theory.py \
          --team "your-team-name" --steps 300000 --out submission.pt
   ```
   `--resume submission.pt` continues a run; `--seed` varies init.

3. **Self-check** before submitting:
   ```bash
   python scripts/evaluate.py submission.pt
   ```
   You should comfortably beat the Random reference before uploading.
   Train and evaluate a null-theory control (this template unchanged) so
   your report can show what your theory added.

4. **Submit** exactly one `submission.pt` per team (your `theory.py`
   travels inside it), through the class platform.

Rules: do not modify anything under `see/` — the tournament runs the
canonical engine regardless; a mismatch is a disqualifying offense. Your
`theory.py` must be deterministic given its inputs and self-contained
(numpy + `see` imports only).
