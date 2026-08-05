"""Strategic Endurance Engine (SEE).

Executable version of the game Gamma from "Strategic Endurance under
Uncertainty" (May 2026), plus training and tournament tooling for the
Game Theory final project.
"""
from .config import (SEEConfig, canonical_config, PLAYERS, IRAN, US,
                     SIGNAL_LEVELS, N_ACTIONS, ACTION_EXIT, ACTION_HOLD)
from .env import StrategicEnduranceEnv, OBS_DIM, other

__version__ = "1.0.0"
