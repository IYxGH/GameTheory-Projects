"""StrategicEnduranceEnv - executable version of the game Gamma.

Implements the multi-stage war of attrition with incomplete information,
observed actions, history-dependent movers, costly signaling and evolving
endurance defined in "Strategic Endurance under Uncertainty" (May 2026),
following the interface sketched in the SEE engine spec.

Key contracts
-------------
* Imperfect information: each player observes its OWN type theta_i and
  endurance e_i, plus the PUBLIC history. The opponent's theta_j / e_j never
  appear in any observation (they do appear in `info["_true_state"]`, which
  exists for tests/debugging only - the tournament runner never passes info
  to agents).
* Reward = exact utility decomposition. Every stage each live player receives
  -(g_i + phi_i); the final stage additionally pays B_i (outlaster) or X_i
  (exiter / simultaneous exit / timeout). Undiscounted episode return
  therefore equals the paper's realized utility u_i exactly - the dense
  reward is a decomposition, not an approximation.
* Action masking: (i) a player outside the active mover set M(h_t) is forced
  to HOLD; (ii) a player whose endurance has dropped to e_i <= 0 is forced to
  EXIT (paper: forced exit).

Action encoding (see config.py): 0..4 = continue with signal level,
5 = EXIT (sigma=0), 6 = HOLD (only when not an active mover).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    ACTION_EXIT,
    ACTION_HOLD,
    HIGH_SIGNAL,
    IRAN,
    N_ACTIONS,
    PLAYERS,
    SEEConfig,
    SIGNAL_LEVELS,
    US,
    canonical_config,
)

OBS_DIM = 16
# Observation vector layout (per stage, per player i; j = opponent):
#  0  t / T_bar
#  1  rho_bar_i                (private)
#  2  m_hat_i                  (private, m_bar normalized to [0,1])
#  3  e_hat_i = e_i / e0_i     (private, clipped to [0, 1.2])
#  4  K_i / 4                  (public: own commitment stock)
#  5  K_j / 4                  (public: opponent commitment stock)
#  6  last sigma_i             (public)
#  7  last sigma_j             (public)
#  8  mediation flag M_t       (public)
#  9  i in M(h_t)              (public)
# 10  j in M(h_t)              (public)
# 11  fraction of past stages with sigma_j >= HIGH_SIGNAL   (public)
# 12  mean past sigma_j        (public)
# 13  mean past sigma_i        (public)
# 14  own cumulative signaling spend / 20                   (private)
# 15  own cumulative effective cost G_i / e0_i              (private)


def other(pid: str) -> str:
    return US if pid == IRAN else IRAN


class StrategicEnduranceEnv:
    """Two-player multi-stage endurance game. Gymnasium-style dict API."""

    def __init__(self, config: Optional[SEEConfig] = None,
                 seed: Optional[int] = None):
        self.cfg = config or canonical_config()
        self._rng = np.random.default_rng(seed)
        self._episode_live = False

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None
              ) -> Tuple[Dict[str, np.ndarray], dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg

        # -- draw private structural types theta_i = (rho_bar, m_bar) ~ p_i
        self.rho: Dict[str, float] = {}
        self.m_bar: Dict[str, float] = {}
        self.m_hat: Dict[str, float] = {}
        self.e0: Dict[str, float] = {}
        self.e: Dict[str, float] = {}
        for pid in PLAYERS:
            pp = cfg.players[pid]
            rho = float(self._rng.beta(pp.rho_a, pp.rho_b))
            mb = pp.m_lo + (pp.m_hi - pp.m_lo) * float(
                self._rng.beta(pp.m_a, pp.m_b))
            m_hat = (mb - pp.m_lo) / (pp.m_hi - pp.m_lo)
            self.rho[pid] = rho
            self.m_bar[pid] = mb
            self.m_hat[pid] = m_hat
            e0 = pp.E0 * (pp.e0_base + pp.e0_rho * rho + pp.e0_m * m_hat)
            self.e0[pid] = e0
            self.e[pid] = e0

        # -- public state
        self.t = 0
        self.K: Dict[str, float] = {p: 0.0 for p in PLAYERS}
        self.last_sigma: Dict[str, float] = {p: 0.0 for p in PLAYERS}
        self.mediation = 0
        self.exited: Dict[str, bool] = {p: False for p in PLAYERS}
        self.public_history: List[dict] = []

        # -- bookkeeping (for exact utility accounting & diagnostics)
        self.G: Dict[str, float] = {p: 0.0 for p in PLAYERS}        # sum g
        self.phi_spend: Dict[str, float] = {p: 0.0 for p in PLAYERS}
        self.sum_sigma: Dict[str, float] = {p: 0.0 for p in PLAYERS}
        self.n_high: Dict[str, int] = {p: 0 for p in PLAYERS}
        self.utility: Dict[str, float] = {p: 0.0 for p in PLAYERS}

        self._active = set(PLAYERS)          # t=0: simultaneous
        self._episode_live = True
        return self._observations(), self._info()

    # ------------------------------------------------------------------ #
    # public queries (used by runner and by student TheorySpec code)     #
    # ------------------------------------------------------------------ #
    def get_active_movers(self) -> List[str]:
        return [p for p in PLAYERS if p in self._active]

    def get_legal_mask(self, pid: str) -> np.ndarray:
        """Boolean mask over the 7 discrete actions."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if not self._episode_live or self.exited[pid]:
            mask[ACTION_HOLD] = True                      # inert
            return mask
        if pid not in self._active:
            mask[ACTION_HOLD] = True                      # forced hold
            return mask
        if self.e[pid] <= 0.0:
            mask[ACTION_EXIT] = True                      # forced exit
            return mask
        mask[:ACTION_EXIT + 1] = True                     # signals + exit
        return mask

    def get_legal_masks(self) -> Dict[str, np.ndarray]:
        return {p: self.get_legal_mask(p) for p in PLAYERS}

    def public_state(self) -> dict:
        """Everything both players can see (for belief modules)."""
        return {
            "t": self.t,
            "T_bar": self.cfg.T_bar,
            "K": dict(self.K),
            "last_sigma": dict(self.last_sigma),
            "mediation": self.mediation,
            "active": sorted(self._active),
            "history": [dict(h) for h in self.public_history],
        }

    # ------------------------------------------------------------------ #
    # dynamics                                                           #
    # ------------------------------------------------------------------ #
    def step(self, actions: Dict[str, int]):
        """Advance one stage. `actions` must contain an entry for every
        player (non-movers must send ACTION_HOLD; the runner enforces this
        via the legal masks)."""
        assert self._episode_live, "call reset() first"
        cfg = self.cfg

        # -- 1. validate & decode actions ------------------------------ #
        sigma_new: Dict[str, float] = {}
        d_exit: Dict[str, bool] = {}
        for pid in PLAYERS:
            a = int(actions[pid])
            mask = self.get_legal_mask(pid)
            if not mask[a]:
                # illegal action: coerce to the first legal one (defensive;
                # the tournament runner also enforces masks agent-side)
                a = int(np.argmax(mask))
            if a == ACTION_HOLD:
                sigma_new[pid] = self.last_sigma[pid]     # posture persists
                d_exit[pid] = False
            elif a == ACTION_EXIT:
                sigma_new[pid] = 0.0                      # de-escalatory exit
                d_exit[pid] = True
            else:
                sigma_new[pid] = SIGNAL_LEVELS[a]
                d_exit[pid] = False
            actions[pid] = a

        # -- 2. flow costs g_i and signaling costs phi_i --------------- #
        rewards = {p: 0.0 for p in PLAYERS}
        g_t: Dict[str, float] = {}
        phi_t: Dict[str, float] = {}
        for pid in PLAYERS:
            j = other(pid)
            pp = cfg.players[pid]
            # raw costs use the CURRENT standing postures (this stage's
            # chosen/held signals), i.e. costs respond to actions at once
            c = pp.c0 + pp.kappa * sigma_new[j]
            a_aud = pp.alpha * self.K[pid]
            r = pp.eta * sigma_new[pid] * sigma_new[j]
            ramp = min(1.0, (self.t + 1) / cfg.ramp_T)   # escalation ramp
            e_hat = max(self.e[pid] / self.e0[pid], 0.0)
            m_t = self.m_bar[pid] * (pp.m_floor + (1 - pp.m_floor) * e_hat)
            g = ramp * (c + a_aud + r) / m_t
            # phi: paid only when a NEW signal is produced this stage
            # (active mover choosing sigma; HOLD maintains posture at no new
            # phi - standing costs flow through K -> a_aud and r instead)
            if actions[pid] < ACTION_EXIT:
                phi = (pp.phi0 * sigma_new[pid] ** 2
                       / (pp.phi_w + (1 - pp.phi_w) * self.rho[pid])
                       / (pp.phi_w + (1 - pp.phi_w) * min(e_hat, 1.0))
                       / (pp.phi_v + (1 - pp.phi_v) * self.m_hat[pid]))
            else:
                phi = 0.0
            g_t[pid] = g
            phi_t[pid] = phi
            rewards[pid] -= (g + phi)
            self.G[pid] += g
            self.phi_spend[pid] += phi

        # -- 3. endurance transition e' = e - g + noise ----------------- #
        for pid in PLAYERS:
            noise = (self._rng.normal(0.0, cfg.e_noise)
                     if cfg.e_noise > 0 else 0.0)
            self.e[pid] = self.e[pid] - g_t[pid] + noise

        # -- 4. resolve exits / terminal values ------------------------- #
        exit_now = [p for p in PLAYERS if d_exit[p]]
        terminated = False
        if exit_now:
            terminated = True
            for pid in PLAYERS:
                if d_exit[pid]:
                    rewards[pid] += self._exit_value(pid, sigma_new)
                else:
                    rewards[pid] += self._outlast_value(pid)
        elif self.t + 1 >= cfg.T_bar:
            # timeout: stalemate treated as simultaneous accommodation
            terminated = True
            for pid in PLAYERS:
                rewards[pid] += self._exit_value(pid, sigma_new)

        # -- 5. public state updates ------------------------------------ #
        self.public_history.append({
            "t": self.t,
            "sigma": dict(sigma_new),
            "action": {p: int(actions[p]) for p in PLAYERS},
            "exit": dict(d_exit),
            "active": sorted(self._active),
            "mediation": self.mediation,
        })
        for pid in PLAYERS:
            self.K[pid] = cfg.delta_K * self.K[pid] + sigma_new[pid]
            self.sum_sigma[pid] += sigma_new[pid]
            if sigma_new[pid] >= HIGH_SIGNAL:
                self.n_high[pid] += 1
            self.last_sigma[pid] = sigma_new[pid]
            self.exited[pid] = self.exited[pid] or d_exit[pid]
            self.utility[pid] += rewards[pid]

        # mediation window Markov chain (public)
        joint = sigma_new[IRAN] + sigma_new[US]
        if self.mediation == 0:
            p_open = (cfg.p_open_calm if joint <= cfg.calm_thresh
                      else cfg.p_open_hot)
            if self._rng.random() < p_open:
                self.mediation = 1
        else:
            if self._rng.random() < cfg.p_close:
                self.mediation = 0

        # -- 6. next active mover set M(h_{t+1}) ------------------------ #
        self.t += 1
        if not terminated:
            hi = {p: sigma_new[p] >= HIGH_SIGNAL for p in PLAYERS}
            if hi[IRAN] != hi[US]:
                escalator = IRAN if hi[IRAN] else US
                self._active = {other(escalator)}     # response turn
            else:
                self._active = set(PLAYERS)           # simultaneous
        else:
            self._episode_live = False
            self._active = set()

        obs = self._observations()
        term = {p: terminated for p in PLAYERS}
        trunc = {p: False for p in PLAYERS}
        return obs, rewards, term, trunc, self._info(g_t, phi_t)

    # ------------------------------------------------------------------ #
    # terminal values (paper Sec. 7 & 9)                                 #
    # ------------------------------------------------------------------ #
    def _exit_value(self, pid: str, sigma_now: Dict[str, float]) -> float:
        pp = self.cfg.players[pid]
        j = other(pid)
        x = (pp.x0 - pp.lam * self.K[pid] - pp.h * sigma_now[j]
             + pp.w_med * self.mediation)
        return float(x)

    def _outlast_value(self, pid: str) -> float:
        pp = self.cfg.players[pid]
        e_hat = float(np.clip(self.e[pid] / self.e0[pid], 0.0, 1.2))
        b = pp.b0 * (pp.b_base + pp.b_e * e_hat + pp.b_rho * self.rho[pid])
        return float(b)

    # ------------------------------------------------------------------ #
    # observations                                                       #
    # ------------------------------------------------------------------ #
    def _observations(self) -> Dict[str, np.ndarray]:
        return {p: self._obs_for(p) for p in PLAYERS}

    def _obs_for(self, pid: str) -> np.ndarray:
        j = other(pid)
        n = max(self.t, 1)
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0] = self.t / self.cfg.T_bar
        o[1] = self.rho[pid]
        o[2] = self.m_hat[pid]
        o[3] = float(np.clip(self.e[pid] / self.e0[pid], 0.0, 1.2))
        o[4] = self.K[pid] / 4.0
        o[5] = self.K[j] / 4.0
        o[6] = self.last_sigma[pid]
        o[7] = self.last_sigma[j]
        o[8] = float(self.mediation)
        o[9] = float(pid in self._active)
        o[10] = float(j in self._active)
        o[11] = self.n_high[j] / n
        o[12] = self.sum_sigma[j] / n
        o[13] = self.sum_sigma[pid] / n
        o[14] = self.phi_spend[pid] / 20.0
        o[15] = self.G[pid] / self.e0[pid]
        return o

    def _info(self, g_t=None, phi_t=None) -> dict:
        info = {
            "t": self.t,
            "active": sorted(self._active),
            "legal_masks": self.get_legal_masks(),
            "public": self.public_state(),
        }
        if g_t is not None:
            info["g"] = g_t
            info["phi"] = phi_t
        # ground truth - for tests and diagnostics ONLY; never shown to agents
        info["_true_state"] = {
            "rho": dict(self.rho), "m_bar": dict(self.m_bar),
            "e": dict(self.e), "e0": dict(self.e0),
            "G": dict(self.G), "utility": dict(self.utility),
        }
        return info
