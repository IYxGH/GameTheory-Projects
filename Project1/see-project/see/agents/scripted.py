"""Scripted baseline agents.

These serve two purposes:
  1. calibration and sanity checks of the environment,
  2. fixed reference opponents on the leaderboard (every submission is also
     scored against them).
"""
from __future__ import annotations

import numpy as np

from ..config import ACTION_EXIT, SIGNAL_LEVELS
from .base import Agent

# obs indices (keep in sync with env.py layout)
T_FRAC, RHO, M_HAT, E_HAT = 0, 1, 2, 3
K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = 4, 5, 6, 7, 8


def _sig_action(level: float) -> int:
    return int(np.argmin([abs(level - s) for s in SIGNAL_LEVELS]))


class RandomAgent(Agent):
    name = "Random"

    def act(self, obs, legal_mask, public_state=None):
        return self.uniform_legal(legal_mask)


class TitForTatAgent(Agent):
    """Mirrors the opponent's last signal; exits when nearly exhausted."""
    name = "TitForTat"

    def act(self, obs, legal_mask, public_state=None):
        if legal_mask[ACTION_EXIT] and obs[E_HAT] < 0.20:
            return ACTION_EXIT
        a = _sig_action(obs[SIG_OPP])
        return a if legal_mask[a] else self.first_legal(legal_mask)


BASELINES = {
    "Random": RandomAgent,
    "TitForTat": TitForTatAgent,
}
