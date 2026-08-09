"""Static HTML leaderboard generator (self-contained, no CDN, offline-safe).

Reads the results.json produced by runner.py and emits a single HTML file:
ranking table with per-role scores and CIs, plus a head-to-head matrix of
mean utilities (row = playing Iran, column = playing U.S.).
"""
from __future__ import annotations

import html
import json


def _color(v: float, lo: float, hi: float) -> str:
    """Red (bad) -> gray -> green (good) on a diverging scale."""
    if hi <= lo:
        return "#f5f5f5"
    x = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    if x < 0.5:
        t = x / 0.5
        r, g, b = 214, int(88 + t * (120 - 88)), 88
        r = int(214 - t * (214 - 128))
        b = int(88 + t * (128 - 88))
    else:
        t = (x - 0.5) / 0.5
        r = int(128 - t * (128 - 46))
        g = int(120 + t * (160 - 120))
        b = int(128 - t * (128 - 96))
    return f"rgb({r},{g},{b})"


def generate(results: dict, out_path: str = "leaderboard.html",
             title: str = "Strategic Endurance — Class Tournament"):
    board = results["leaderboard"]
    pairs = results["pairs"]
    names = [r["name"] for r in board]

    h2h = {(p["iran"], p["us"]): p for p in pairs}
    all_u = [p["u_I"] for p in pairs] + [p["u_U"] for p in pairs]
    lo, hi = min(all_u), max(all_u)

    rows = []
    for r in board:
        cls = "baseline" if r["baseline"] else "student"
        medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(
            r["rank"], "") if not r["baseline"] else ""
        rows.append(f"""
        <tr class="{cls}">
          <td class="rank">{r['rank']}</td>
          <td class="name">{medal} {html.escape(r['name'])}</td>
          <td class="score">{r['score']:.1f}
              <span class="ci">[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]</span></td>
          <td>{r['score_as_iran']:.1f}</td>
          <td>{r['score_as_us']:.1f}</td>
          <td class="eps">{r['episodes']}</td>
        </tr>""")

    matrix_head = "".join(
        f"<th><div class='rot'><span>{html.escape(n)}</span></div></th>"
        for n in names)
    matrix_rows = []
    for ni in names:
        cells = []
        for nu in names:
            if ni == nu:
                cells.append("<td class='diag'>—</td>")
                continue
            p = h2h.get((ni, nu))
            if p is None:
                cells.append("<td>·</td>")
                continue
            cells.append(
                f"<td style='background:{_color(p['u_I'], lo, hi)}' "
                f"title='I: {html.escape(ni)} vs U: {html.escape(nu)}\n"
                f"u_I={p['u_I']:.1f}  u_U={p['u_U']:.1f}  len={p['len']:.1f}'>"
                f"{p['u_I']:.0f}</td>")
        matrix_rows.append(
            f"<tr><th class='rowh'>{html.escape(ni)}</th>"
            + "".join(cells) + "</tr>")

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{ --ink:#1a1a2e; --mut:#6b7280; --line:#e5e7eb; --acc:#0f4c81; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; color:var(--ink);
         max-width:1080px; margin:2rem auto; padding:0 1rem; }}
  h1 {{ font-size:1.6rem; margin-bottom:.2rem; }}
  .sub {{ color:var(--mut); font-size:.9rem; margin-bottom:1.6rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
  th,td {{ padding:.45rem .6rem; border-bottom:1px solid var(--line);
           text-align:left; }}
  thead th {{ font-size:.78rem; text-transform:uppercase;
              letter-spacing:.06em; color:var(--mut);
              border-bottom:2px solid var(--ink); }}
  tr.baseline td {{ color:var(--mut); font-style:italic; }}
  td.rank {{ font-weight:700; width:3rem; }}
  td.score {{ font-weight:600; }}
  .ci {{ color:var(--mut); font-weight:400; font-size:.8rem; }}
  td.eps {{ color:var(--mut); }}
  h2 {{ margin-top:2.4rem; font-size:1.15rem; }}
  .matrix td, .matrix th {{ text-align:center; font-size:.78rem;
      padding:.3rem .35rem; border:1px solid #fff; color:#fff; }}
  .matrix td.diag {{ background:#f3f4f6; color:var(--mut); }}
  .matrix th {{ color:var(--ink); background:#fff; }}
  .matrix .rowh {{ text-align:right; font-size:.78rem; max-width:11rem;
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .rot {{ writing-mode:vertical-rl; transform:rotate(200deg);
      transform:rotate(180deg); max-height:9rem; }}
  .note {{ color:var(--mut); font-size:.85rem; margin-top:.8rem; }}
  .pill {{ display:inline-block; background:#eef2f7; color:var(--acc);
      border-radius:99px; padding:.1rem .6rem; font-size:.78rem;
      margin-right:.4rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">
  <span class="pill">generated {results['generated']}</span>
  <span class="pill">{results['episodes_per_side']} episodes / side /
      pairing</span>
  <span class="pill">seed base {results['seed0']}</span>
  <span class="pill">{results['elapsed_sec']}s</span>
</div>

<table>
  <thead><tr><th>#</th><th>Agent</th><th>Score (mean u)</th>
    <th>as Iran</th><th>as U.S.</th><th>Episodes</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<p class="note">Score = mean raw utility u<sub>i</sub> over all pairings,
both roles, common seeds. Brackets: 95% bootstrap CI. Baselines
(italic) are fixed reference agents, not submissions.</p>

<h2>Head-to-head: mean u<sub>I</sub> (row plays Iran, column plays U.S.)</h2>
<div style="overflow-x:auto">
<table class="matrix">
  <tr><th></th>{matrix_head}</tr>
  {''.join(matrix_rows)}
</table></div>
<p class="note">Hover a cell for both utilities and mean episode length.
Green = good outcome for the row (Iran-side) agent.</p>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(page)
    print(f"leaderboard -> {out_path}")


def main(results_path: str = "results.json",
         out_path: str = "leaderboard.html"):
    with open(results_path) as f:
        results = json.load(f)
    generate(results, out_path)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
