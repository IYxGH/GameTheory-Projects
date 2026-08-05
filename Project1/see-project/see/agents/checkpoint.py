"""CheckpointAgent - loads a submitted .pt file for tournament play.

A submission contains: both role networks (fixed architecture), the
embedded theory.py source (needed to recompute the extra belief features
at evaluation time), and training metadata for auditing.

SECURITY NOTE (instructor): loading a checkpoint executes the embedded
theory source and unpickles tensors. Run tournaments in an isolated
environment (container/VM), exactly as you would for any code submission.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from ..config import N_ACTIONS
from ..env import OBS_DIM
from .base import Agent
from .policy_net import SEEPolicy


def pick_device() -> torch.device:
    """SEE_DEVICE env wins; otherwise CUDA when available, else CPU."""
    want = os.environ.get("SEE_DEVICE", "").strip()
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CheckpointAgent(Agent):
    def __init__(self, path: str, sample: bool = True,
                 device: Optional[str] = None):
        from ..training.theory_api import theory_from_source
        # checkpoints contain only tensors/primitives, so the safe
        # weights-only loader suffices (and is the torch>=2.6 default)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        assert ck.get("obs_dim") == OBS_DIM and \
            ck.get("n_actions") == N_ACTIONS, \
            f"{path}: incompatible checkpoint (obs/action dims)"
        self.device = torch.device(device) if device else pick_device()
        self.name = str(ck.get("team", "anonymous"))
        self.extra_dim = int(ck.get("extra_dim", 0))
        self.theory = theory_from_source(ck.get("theory_source", ""))
        assert self.theory.extra_feature_dim == self.extra_dim, \
            f"{path}: theory feature dim mismatch"
        self.nets = {}
        for pid, sd in ck["nets"].items():
            net = SEEPolicy(OBS_DIM, N_ACTIONS, self.extra_dim)
            net.load_state_dict(sd)
            net.to(self.device)
            net.eval()
            self.nets[pid] = net
        self.sample = sample
        self.meta = ck.get("train_meta", {})

    def reset(self, player_id: str, seed: Optional[int] = None):
        super().reset(player_id, seed)
        self.hidden = None
        # the sampling generator must live on the same device as the policy
        self.gen = torch.Generator(device=self.device) \
            if self.device.type == "cuda" else torch.Generator()
        self.gen.manual_seed(int(seed) if seed is not None else 0)
        self.theory.on_episode_start(player_id)

    def act(self, obs: np.ndarray, legal_mask: np.ndarray,
            public_state: Optional[dict] = None) -> int:
        x = obs
        if self.extra_dim:
            extra = np.asarray(
                self.theory.extra_features(self.player_id, obs,
                                           public_state or {}),
                dtype=np.float32)
            extra = np.nan_to_num(extra, nan=0.0, posinf=1e6, neginf=-1e6)
            x = np.concatenate([obs, extra])
        a, _, _, self.hidden = self.nets[self.player_id].step(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(legal_mask), self.hidden,
            sample=self.sample, generator=self.gen)
        return a
