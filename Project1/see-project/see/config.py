"""Canonical instantiation of the game Gamma from
"Strategic Endurance under Uncertainty" (May 2026).

This file is the single source of truth for the *physics* of the
confrontation: every functional form left abstract in the paper is given a
concrete parametric form here, chosen so that all assumptions of the paper
hold (signs of derivatives, single-crossing condition on signaling cost).

Students MUST NOT edit this file. Their theory enters through
`submission_template/theory.py` (belief features + reward shaping), never
through the world itself. The tournament is always run with these canonical
parameters, so every agent competes in exactly the same game.

Mapping to the paper's notation is given field-by-field below.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List

IRAN = "I"
US = "U"
PLAYERS = (IRAN, US)

# Discrete signal ladder sigma in [0,1]: restraint -> maximal escalation /
# public commitment. (The paper allows continuous sigma; a 5-point ladder
# keeps the analytic part of the project tractable and the policy head small.)
SIGNAL_LEVELS: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]
HIGH_SIGNAL = 0.75  # threshold at which a signal triggers a response turn

# Action encoding (7 discrete actions):
#   0..4 : continue (d=C) with signal SIGNAL_LEVELS[a]
#   5    : EXIT (d=E), carries sigma=0 (de-escalatory exit)
#   6    : HOLD - only legal (and forced) when the player is NOT in the
#          active mover set M(h_t); last signal persists, d=C implied.
N_ACTIONS = 7
ACTION_EXIT = 5
ACTION_HOLD = 6


@dataclass
class PlayerParams:
    """Per-player parameters. Asymmetry between I and U lives here."""

    # ---- prior over structural type theta_i = (rho_bar, m_bar) ----------
    # rho_bar ~ Beta(rho_a, rho_b) on [0,1]                     (resolve)
    rho_a: float = 3.0
    rho_b: float = 3.0
    # m_bar ~ m_lo + (m_hi - m_lo) * Beta(m_a, m_b)   (cost-management cap.)
    m_a: float = 2.0
    m_b: float = 2.0
    m_lo: float = 0.7
    m_hi: float = 1.6

    # ---- initial endurance e0_i(theta_i) ---------------------------------
    # e0 = E0 * (e0_base + e0_rho * rho_bar + e0_m * m_hat)
    E0: float = 100.0
    e0_base: float = 0.50
    e0_rho: float = 0.35
    e0_m: float = 0.15

    # ---- raw flow costs (paper Sec. 5) -----------------------------------
    # material/economic: c_i = c0 + kappa * sigma_j(last)
    c0: float = 1.8
    kappa: float = 4.0
    # audience cost:     a_i = alpha * K_i(t)   (K = own commitment stock)
    alpha: float = 1.0
    # escalation risk:   r_i = eta * sigma_i(last) * sigma_j(last)
    eta: float = 3.0
    # capacity:          m_i(t) = m_bar * (m_floor + (1-m_floor) * e_hat)
    m_floor: float = 0.35

    # ---- signaling cost phi (paper Sec. 6) -------------------------------
    # phi = phi0 * sigma^2 / [(w+ (1-w) rho)(w + (1-w) e_hat)(v + (1-v) m_hat)]
    phi0: float = 1.2
    phi_w: float = 0.40   # weight floor for rho and e_hat
    phi_v: float = 0.50   # weight floor for m_hat

    # ---- value of outlasting B (paper Sec. 7) ----------------------------
    # B = b0 * (b_base + b_e * e_hat + b_rho * rho_bar)
    b0: float = 110.0
    b_base: float = 0.50
    b_e: float = 0.30
    b_rho: float = 0.20

    # ---- exit value X (paper Sec. 7) --------------------------------------
    # X = x0 - lam * K_i(t) - h * sigma_j(last) + w_med * M_t
    x0: float = 46.0
    lam: float = 5.0
    h: float = 8.0
    w_med: float = 20.0


@dataclass
class SEEConfig:
    """Full canonical configuration of the Strategic Endurance Engine."""

    # horizon T_bar (paper: t = 0..T_bar)
    T_bar: int = 40

    # commitment stock: K_i(t+1) = delta_K * K_i(t) + sigma_i(t)
    delta_K: float = 0.75

    # escalation ramp: raw flow costs scale by min(1, (t+1)/ramp_T).
    # Early stages are structurally cheap (confrontations heat up
    # gradually), which preserves the option value of observing the
    # opponent before deciding - without it, immediate exit can become
    # degenerately dominant for the pressured side.
    ramp_T: int = 4

    # mediation window M_t (public):
    #   closed -> opens w.p. p_open_calm if sigma_I+sigma_U <= calm_thresh,
    #             else w.p. p_open_hot
    #   open   -> closes w.p. p_close
    p_open_calm: float = 0.35
    p_open_hot: float = 0.05
    p_close: float = 0.25
    calm_thresh: float = 0.50

    # endurance shock xi ~ N(0, e_noise^2) (keeps the opponent's filtering
    # problem non-degenerate; set 0.0 for a deterministic classroom variant)
    e_noise: float = 0.5

    # timeout: at t = T_bar both players receive X_i(T_bar, h) (stalemate is
    # treated as a de-facto mutual accommodation)
    players: Dict[str, PlayerParams] = field(default_factory=lambda: {
        IRAN: PlayerParams(
            # Iran: resolve-heavy prior, weaker cost management
            rho_a=5.0, rho_b=2.0,          # E[rho] ~ 0.71
            m_a=2.0, m_b=2.0, m_lo=0.8, m_hi=1.7,
            c0=1.4, kappa=3.0,             # suffers more from U.S. pressure
            alpha=0.8,                     # lower audience sensitivity
            eta=2.5,
            b0=175.0, x0=26.0,
        ),
        US: PlayerParams(
            # U.S.: diffuse resolve, stronger cost management
            rho_a=3.0, rho_b=3.0,          # E[rho] = 0.50
            m_a=3.0, m_b=2.0, m_lo=0.9, m_hi=1.8,
            c0=1.8, kappa=2.5,             # less exposed to raw pressure
            alpha=1.3,                     # higher audience sensitivity
            eta=2.5,
            b0=175.0, x0=28.0,
        ),
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SEEConfig":
        d = copy.deepcopy(d)
        players = d.pop("players", {})
        cfg = cls(**{k: v for k, v in d.items() if k != "players"})
        for pid, pd in players.items():
            base = asdict(cfg.players[pid])
            base.update(pd)
            cfg.players[pid] = PlayerParams(**base)
        return cfg


def canonical_config() -> SEEConfig:
    """The tournament configuration. Do not modify for the competition."""
    return SEEConfig()
