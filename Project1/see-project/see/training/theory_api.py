"""TheorySpec - the interface through which YOUR game theory enters the agent.

This is the only file students write. A TheorySpec supplies:

1.  extra_features(...)  -  BELIEFS MADE EXPLICIT.
    Called at every stage (training AND evaluation). Return a fixed-length
    vector of numbers computed from your own observation and the PUBLIC
    history. WHAT to compute is your Section 4 analysis (see the project
    manual); this docstring only defines the interface, not the answer.
    These are appended to the network input, so the policy can condition
    directly on your theory's sufficient statistics instead of having to
    rediscover them from raw history.

2.  shaping(...)  -  THE OPTIMIZATION FORMULA.
    Called during TRAINING ONLY. Return an additional reward added to the
    environment reward at each step. Use it to encode the conclusions of
    your Section 4 analysis as a learning signal; WHAT that signal should
    be is your work, not something stated here. WARNING: the tournament
    scores RAW utility. Shaping that distorts the objective away from u_i
    will train an agent that optimizes the wrong thing. Potential-based
    shaping F = gamma * Phi(s') - Phi(s), with Phi a function you choose,
    provably preserves the optimal policy (Ng, Harada & Russell 1999).

Rules of the game:
  * You may use anything visible in `obs` and `public_state` (own private
    state + public history). You have NO access to the opponent's theta/e.
  * `extra_feature_dim` must be a constant; features must be finite floats.
  * Keep it deterministic given (obs, public_state) - it runs at evaluation.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Optional

import numpy as np


class TheorySpec:
    """Default: no extra features, no shaping (pure environment reward)."""

    name: str = "null-theory"
    extra_feature_dim: int = 0

    def on_episode_start(self, player_id: str) -> None:
        """Reset any per-episode state (filters, caches)."""

    def extra_features(self, player_id: str, obs: np.ndarray,
                       public_state: dict) -> np.ndarray:
        return np.zeros(self.extra_feature_dim, dtype=np.float32)

    def shaping(self, player_id: str, obs: np.ndarray, action: int,
                env_reward: float, next_obs: np.ndarray, next_public: dict,
                terminated: bool) -> float:
        return 0.0


def load_theory(path: str) -> tuple[TheorySpec, str]:
    """Load the (single) TheorySpec subclass from a student file.

    Returns (instance, source_text). The source is embedded into the
    checkpoint so the instructor can rebuild the exact feature pipeline at
    evaluation time and audit what was submitted.
    """
    p = pathlib.Path(path)
    source = p.read_text()
    spec = importlib.util.spec_from_file_location("student_theory", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [v for v in vars(mod).values()
               if isinstance(v, type) and issubclass(v, TheorySpec)
               and v is not TheorySpec]
    if not classes:
        return TheorySpec(), source
    if len(classes) > 1:
        raise ValueError(f"{path} defines {len(classes)} TheorySpec "
                         "subclasses; define exactly one.")
    return classes[0](), source


def theory_from_source(source: str) -> TheorySpec:
    """Rebuild a TheorySpec from source text embedded in a checkpoint."""
    ns: dict = {"TheorySpec": TheorySpec, "np": np, "numpy": np}
    exec(compile(source, "<checkpoint-theory>", "exec"), ns)
    classes = [v for v in ns.values()
               if isinstance(v, type) and issubclass(v, TheorySpec)
               and v is not TheorySpec]
    return classes[0]() if classes else TheorySpec()
