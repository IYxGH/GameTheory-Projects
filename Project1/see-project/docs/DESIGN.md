# SEE Design Document — Canonical Instantiation of Γ

**Audience:** instructor (and curious students after the competition).
**Status:** v1.0 — the tournament physics. Any change here is a new game version.

This document fixes every functional form that the paper *Strategic
Endurance under Uncertainty* (May 2026) leaves abstract, explains why each
form satisfies the paper's assumptions, and documents the deliberate
modeling choices made to turn a conceptual framework into a runnable,
non-degenerate competition environment.

---

## 1. Why a canonical instantiation exists

The paper defines the game
Γ = ⟨N, Θ, p, E, T, H, M, A, Σ, D, S, g, G, φ, B, X, µ, τ, u⟩
only up to signs and derivative conditions. An executable engine needs
numbers. More importantly, the **competition needs one shared world**: if
every team defined its own cost functions, two submitted agents would have
been trained on *different games*, and a head-to-head match would have no
canonical dynamics — the leaderboard would compare apples to oranges.

Therefore: the environment (this document, `see/config.py`, `see/env.py`)
is fixed and public. Student theory enters through the *solution*, not the
world: belief features and reward shaping in their `theory.py`, informed by
their Section 4 theoretical analysis (see the project manual).

## 2. Players, types, priors

N = {I, U}. Structural type θᵢ = (ρ̄ᵢ, m̄ᵢ), drawn once at t = 0,
independent across players (as in the paper's joint prior p = p_I·p_U):

| | ρ̄ (resolve) | m̄ (cost management) |
|---|---|---|
| Iran | Beta(5, 2) on [0,1] — E[ρ̄]≈0.71 | 0.8 + 0.9·Beta(2,2) — E≈1.25 |
| U.S. | Beta(3, 3) on [0,1] — E[ρ̄]=0.50 | 0.9 + 0.9·Beta(3,2) — E≈1.44 |

The asymmetry operationalizes Section 2 of the paper: Iran's prior is
resolve-heavy (endurance-as-absorption), the U.S. prior gives stronger and
more reliable cost-management capacity (endurance-as-sustained-pressure)
but diffuse resolve. `m̂ᵢ ∈ [0,1]` denotes m̄ normalized within its range.

**Initial endurance** (paper: e⁰ᵢ = e⁰ᵢ(θᵢ)):
e⁰ᵢ = 100·(0.50 + 0.35·ρ̄ᵢ + 0.15·m̂ᵢ) ∈ [50, 100].

## 3. Stages, actions, signals

t = 0, 1, …, T̄−1 with **T̄ = 40**.

Complete action aᵢᵗ = (σᵢᵗ, dᵢᵗ). The signal is a five-point ladder
σ ∈ {0, 0.25, 0.5, 0.75, 1.0} (restraint → maximal escalation/commitment).
*Deviation from the paper:* the paper allows continuous σ; discretization
keeps the analytic part of the project tractable (finite strategy space)
and the policy head small. dᵢᵗ ∈ {C, E} as in the paper; the exit action
carries σ = 0 (exit is de-escalatory by construction).

**Action encoding** (7 discrete actions): 0–4 = continue with signal level;
5 = EXIT; 6 = HOLD (see mover rule).

## 4. History-dependent mover sets M(hᵗ)

- t = 0: simultaneous, M = {I, U}.
- If **exactly one** player produced a high signal (σ ≥ 0.75) at t−1, the
  next stage is a **response turn**: only the *other* player moves. The
  escalator is forced to HOLD — its posture persists publicly, it cannot
  exit that stage (it is briefly locked into its own escalation: a concrete
  form of the credibility–flexibility trap).
- Otherwise (both high, both low): simultaneous.

This is the paper's "at some histories one player moves and the other
observes and responds" made mechanical and deterministic (auditable by
students from the public history alone).

**HOLD semantics:** the held player's last σ remains the standing posture
(enters both players' cost terms), d = C is implied, and **no new signaling
cost φ is charged** — standing-commitment costs flow instead through the
commitment stock K → audience costs (Sec. 5). Both players accrue flow
costs every stage regardless of who moves.

## 5. Effective cost of continuing (paper Sec. 5)

Public **commitment stock** (drives audience costs and exit erosion):
Kᵢᵗ⁺¹ = 0.75·Kᵢᵗ + σᵢᵗ, K⁰ = 0 (so K ≤ 4).

Raw flow components for player i (σ̃ = current standing signals):

- material/economic: cᵢ = c0ᵢ + κᵢ·σ̃ⱼ — pressure from the *opponent's*
  posture. Iran: c0 = 1.4, κ = 3.0 (more exposed); U.S.: c0 = 1.8, κ = 2.5.
- domestic audience: aᵢᵃᵘᵈ = αᵢ·Kᵢᵗ — the cost of one's own standing
  public commitments. Iran α = 0.8; U.S. α = 1.3 (more audience-sensitive).
- escalation risk: rᵢ = ηᵢ·σ̃ᵢ·σ̃ⱼ, η = 2.5 — bilinear: risk needs two.

All three raw components are scaled by the **escalation ramp**
min(1, (t+1)/4): the first stages of a confrontation are structurally
cheap. This preserves the option value of observing the opponent for a few
rounds before deciding — without it, immediate exit is degenerately
dominant for the pressured side (found in calibration, Sec. 11).

**Cost-management capacity** (paper: mᵢᵗ = mᵢ(θᵢ, eᵢᵗ, hᵗ) > 0):
mᵢᵗ = m̄ᵢ·(0.35 + 0.65·êᵢᵗ), ê = max(e/e⁰, 0) — capacity degrades as
endurance is spent, producing the paper's compounding fragility (the same
raw pressure hurts more when you are already worn down).

**Effective flow cost and endurance law:**
gᵢᵗ = (cᵢ + aᵢᵃᵘᵈ + rᵢ)/mᵢᵗ,  eᵢᵗ⁺¹ = eᵢᵗ − gᵢᵗ + ξᵗ, ξ ~ N(0, 0.5²).

The small shock ξ (an extension of the deterministic SEE draft) keeps the
opponent's filtering problem non-degenerate: even a player who knows the
cost law cannot invert endurance exactly from public history. Set
`e_noise = 0` in config for a deterministic classroom variant.

Note the useful identity: accumulated effective cost Gᵢ ≈ e⁰ᵢ − eᵢ (up to
noise) — "endurance spent" and "effective cost borne" are the same object.

## 6. Costly signaling (paper Sec. 6)

φᵢ(σ) = 1.2·σ² / [(0.4 + 0.6·ρ̄ᵢ)(0.4 + 0.6·min(êᵢ,1))(0.5 + 0.5·m̂ᵢ)]

All paper conditions hold by construction:
∂φ/∂σ > 0 (stronger signals cost more); ∂φ/∂ê < 0, ∂φ/∂ρ̄ < 0,
∂φ/∂m̂ < 0 (the same signal is cheaper for stronger types); and the
**single-crossing condition** ∂²φ/∂σ∂ê < 0, ∂²φ/∂σ∂ρ̄ < 0 (marginal
signal cost falls with strength), which is what allows strong signals to be
informative in equilibrium — weak types can only imitate at higher cost.
`tests/test_env.py::test_single_crossing_of_signaling_cost` verifies this
numerically. φ is charged when a new signal is *produced* (active mover
choosing σ); the paper's φ^commit component is carried by K → a_aud and
by the exit penalty λ·K below.

## 7. Outlasting and exit values (paper Sec. 7)

**Value of outlasting** (paid to i at the stage the opponent exits):
Bᵢ = 175·(0.50 + 0.30·êᵢ + 0.20·ρ̄ᵢ) — increasing in current endurance
and baseline resolve, as required (∂B/∂e > 0, ∂B/∂ρ̄ > 0). Winning
exhausted is worth much less than winning strong. b0 = 175 is a calibrated
choice: the winner's prize must clearly dominate the *net* surplus
destroyed by a fought war, or exiting immediately outperforms fighting
even for likely winners and the game degenerates (Sec. 11).

**Dynamic exit value:**
Xᵢ(t, hᵗ) = x0ᵢ − 5·Kᵢᵗ − 8·σ̃ⱼ + 20·Mᵗ, with x0 = 26 (Iran), 28 (U.S.).

Each term implements a mechanism from the paper: −λK is **self-trapping**
(your own accumulated public commitments make exit politically costly);
−h·σ̃ⱼ is **humiliation** (exiting under the opponent's high posture is
worse — face matters); +w·Mᵗ is **constructed exit** (an open mediation
window makes leaving politically cheaper).

**Mediation window** Mᵗ ∈ {0,1} (public Markov chain): closed → opens with
prob 0.35 if joint posture is calm (σ̃ᵢ+σ̃ⱼ ≤ 0.5), else 0.05; open →
closes with prob 0.25. Third parties matter, and restraint invites them.

## 8. Termination and utilities (paper Sec. 9)

Exactly the paper's uᵢ:

- i exits first (τᵢ < τⱼ): uᵢ = Xᵢ(τᵢ) − Gᵢ(τᵢ) − Σφᵢ, and the survivor
  j gets uⱼ = Bⱼ(e^τᵢ) − Gⱼ(τᵢ) − Σφⱼ (episode ends at the first exit).
- Both exit at the same stage: both receive X.
- eᵢ ≤ 0 at the start of a stage: i is **forced** to exit (legal-action
  mask collapses to {EXIT}).
- Timeout t = T̄: both receive Xᵢ(T̄, h) — stalemate is treated as de
  facto mutual accommodation (and both sides' K erode their X, so drifting
  to timeout with heavy commitments is a bad outcome for both).

**Reward = exact utility decomposition.** The engine emits, at every stage,
rᵢᵗ = −(gᵢᵗ + φᵢᵗ), plus the terminal B or X. The undiscounted episode
return equals uᵢ *exactly* (verified by
`test_return_equals_paper_utility`). The "reward shaping" of the SEE draft
spec is thus not an approximation baked into the engine; dense guidance
beyond this is the students' job (their TheorySpec, training-time only).

## 9. The RL mapping

Per player the game is a POMDP: state = (θ_I, θ_U, e_I, e_U, K, M, t,
movers), observation = own type + own endurance + public history (the
paper's information set — beliefs µⱼ must be *computed by the agent*, which
is why the standard policy is recurrent and why TheorySpec exists).
Observation layout and action encoding are documented in `see/env.py`.

**Standard architecture (fixed for fairness):** obs(16 + extra) → Linear 64
→ tanh → GRU 64 → {7 masked logits, value}. Separate parameter sets per
role (the game is asymmetric); one training run produces the pair;
submissions are checkpoints of this pair plus the embedded theory source.

**Self-play:** PPO (clip 0.2, γ 0.995, GAE 0.95) mixing pure self-play
(35%) with play against frozen snapshots and scripted baselines (65%).
The heavy scripted share is deliberate: pure self-play in this game has a
strong degenerate attractor — "the ex-ante weaker side concedes at t=0" —
because against a twin that never exits, instant concession is a best
response and the gradient toward type-conditional fighting vanishes.
Diverse scripted opponents (who fight, fold, and pressure differently)
keep that gradient alive; expect opponent- and type-conditional behavior
after ~250–350k steps. The result is an *empirical approximate
equilibrium*, not a certified PBE — the honest framing for students:
theory tells you what the equilibrium must look like (cutoff structure,
separating signals); self-play searches near it; the tournament tests
robustness against everyone else's approximation.

## 10. Competition protocol

1. Students edit `theory.py` (belief features + shaping), run
   `scripts/train.py` locally (~15–40 min CPU), sanity-check with
   `scripts/evaluate.py`, submit one `.pt`.
2. Instructor drops all `.pt` files into `submissions/` and runs
   `scripts/run_tournament.py`.
3. Every entrant plays every other entrant and all five scripted baselines,
   **in both roles**, on a **common seed list** (identical type draws,
   shocks, and mediation luck across all pairings — differences in outcome
   are differences in strategy). Score = mean raw utility across both
   roles, with bootstrap CIs; per-role scores shown separately because the
   game is asymmetric. Output: `results.json` + self-contained
   `leaderboard.html`.
4. **Anti-overfitting:** publish a practice seed base; run the official
   final with a different, undisclosed `--seed0`.
5. **Security:** checkpoints embed and execute the submission's feature
   code at load time — run the tournament in a container/VM, as with any
   code submission.

## 11. Calibration evidence (v1.0)

> Note: the public reference pool was later reduced to Random and
> TitForTat. The original calibration below also exercised three now-removed
> scripted archetypes (an always-escalate "Hawk", an always-fold "Dove",
> and a Bayesian-threshold agent); their behavior is described here only as
> historical evidence that the game is non-degenerate. If you reintroduce
> such archetypes to `see/agents/scripted.py` they are picked up by the
> trainer and tournament automatically.

From `scripts/calibrate.py` (60 episodes/pairing): no dominant pure
strategy — an always-escalate agent exploits passive agents but
self-destructs in the mirror match; an always-fold agent is safe but
exploited; a Bayesian-threshold reference folds cheaply against committed
aggressors (correct best response) and pressures weaker types; mirror
matches show type-dependent outcomes. Median episode length ≈ 6–12 stages,
patient symmetric matchups run to 30+, inside T̄ = 40.

Three degeneracies were found and fixed during calibration — watch for
them if you retune:

1. **Immediate-exit dominance** (X too generous vs. waiting): fixed by
   lowering x0 (46→26/28) and adding the escalation ramp (Sec. 5).
2. **War destroys the prize** (B − accumulated G below X even for likely
   winners, so *nobody* should fight): fixed by raising b0 (110→175),
   which restores type-conditional incentives — likely winners fight,
   likely losers fold.
3. **Self-play concession collapse** (one side learns to exit at t≈0 and
   the other never exits; bistable — it flipped sides across seeds): an
   equilibrium-selection pathology of pure self-play, not of the physics.
   Fixed in the *trainer* (scripted-opponent share 65%, entropy ≥0.02
   early). Verified escape: after ~330k steps the benchmark agent
   exploits passive opponents in both roles, refuses mutual destruction
   against Hawk, and plays cautious attrition against fighters.

Re-run `scripts/calibrate.py` plus a ~300k-step training probe after any
parameter change.

## 12. Versioning

The tournament game is `see.__version__` = **1.0.0** with the parameters in
`see/config.py`. Changing any parameter changes the equilibrium — bump the
version and re-announce to the class if you retune.
