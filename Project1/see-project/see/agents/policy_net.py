"""Standard policy architecture - FIXED for the whole class.

Every submission is a checkpoint of exactly this network (one per role).
Fairness of the competition rests on this: all teams train the same
architecture with the same trainer; what differs is the *theory* they feed
it (extra belief features + reward shaping in theory.py) and their training
budget/seeds. Do not modify.

obs (16) ++ theory extra features (K)  ->  Linear(64) tanh  ->  GRU(64)
                                       ->  policy logits (7) & value (1)

The GRU is what lets the policy carry an *implicit posterior* over the
opponent's hidden type and endurance across stages (the mu_j^t of the
paper); students' extra features make parts of that posterior explicit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN = 64


class SEEPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, extra_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.extra_dim = extra_dim
        in_dim = obs_dim + extra_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.Tanh())
        self.gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        self.pi = nn.Linear(HIDDEN, n_actions)
        self.v = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x: (B, T, in_dim) -> logits (B,T,A), value (B,T), h'."""
        z = self.encoder(x)
        out, h = self.gru(z, h)
        return self.pi(out), self.v(out).squeeze(-1), h

    @torch.no_grad()
    def step(self, x_t, mask_t, h: torch.Tensor | None,
             sample: bool = True,
             generator: torch.Generator | None = None):
        """Single-timestep action selection with legal-action masking.

        x_t: (in_dim,), mask_t: bool (A,) — numpy or tensor; they are
        moved to whatever device this module lives on (CPU or CUDA), so
        the same code path serves CPU training and GPU evaluation.
        Returns (action, logp, value, h').
        """
        dev = next(self.parameters()).device
        x = torch.as_tensor(x_t, dtype=torch.float32).to(dev)
        mask = torch.as_tensor(mask_t, dtype=torch.bool).to(dev)
        logits, value, h = self.forward(x.view(1, 1, -1), h)
        logits = logits[0, 0].masked_fill(~mask, -1e9)
        probs = torch.softmax(logits, dim=-1)
        if sample:
            a = int(torch.multinomial(probs, 1, generator=generator).item())
        else:
            a = int(torch.argmax(probs).item())
        logp = float(torch.log(probs[a] + 1e-12).item())
        return a, logp, float(value[0, 0].item()), h


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return torch.where(masks, logits, torch.full_like(logits, -1e9))
