"""Agent interface used by the tournament runner and the trainer.

An agent sees ONLY:
  * its own observation vector (see env.OBS_DIM layout),
  * the legal-action mask,
  * the public state (history of observed actions, commitment stocks,
    mediation flag) - i.e. exactly the information set of the paper.

It never receives the opponent's type or endurance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Agent:
    """Base class. Subclass and override `act`."""

    name: str = "agent"

    def reset(self, player_id: str, seed: Optional[int] = None) -> None:
        """Called at the start of every episode with the role assignment."""
        self.player_id = player_id
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, legal_mask: np.ndarray,
            public_state: Optional[dict] = None) -> int:
        raise NotImplementedError

    # convenience -------------------------------------------------------- #
    @staticmethod
    def first_legal(legal_mask: np.ndarray) -> int:
        return int(np.argmax(legal_mask))

    def uniform_legal(self, legal_mask: np.ndarray) -> int:
        legal = np.flatnonzero(legal_mask)
        return int(self.rng.choice(legal))
