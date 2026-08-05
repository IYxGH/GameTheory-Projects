"""Recurrent PPO self-play trainer for the Strategic Endurance Engine.

One training run learns a PAIR of policies (Iran-side and US-side, separate
parameters - the game is asymmetric). Each iteration:

  1. collect episodes:
       - with prob `p_self`  : current policy pair plays itself
         (both seats collect gradient data);
       - otherwise           : current pair plays one seat against a frozen
         opponent sampled from {past checkpoints} U {scripted baselines}
         (only the current seat collects data).
     The frozen pool is what pushes self-play toward robust, approximately
     equilibrium behavior instead of chasing its own tail.
  2. GAE advantages per episode (gamma, lam), on shaped rewards
     (env reward + TheorySpec.shaping); raw utility is logged separately.
  3. PPO clipped update per role over minibatches of whole episodes
     (GRU BPTT across each episode).

Designed to run on a student laptop CPU in ~15-40 minutes with defaults.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..agents.policy_net import SEEPolicy, masked_logits
from ..agents.scripted import BASELINES
from ..config import IRAN, N_ACTIONS, PLAYERS, US
from ..env import OBS_DIM, StrategicEnduranceEnv
from .theory_api import TheorySpec


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    steps_per_iter: int = 4_096
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch_episodes: int = 32
    entropy_coef: float = 0.025
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.02       # utilities are O(100)
    # Opponent mix. Pure self-play in this game has a strong degenerate
    # basin ("the ex-ante weaker side concedes at t=0"), because against a
    # twin that never exits, conceding immediately is a best response and
    # the gradient toward type-conditional fighting vanishes. A heavy dose
    # of scripted opponents (which behave differently from the learner and
    # from each other) plus entropy keeps that gradient alive. See
    # docs/DESIGN.md Sec. 9. If you add more scripted archetypes to
    # see/agents/scripted.py they are picked up here automatically.
    p_self: float = 0.35             # fraction of pure self-play episodes
    p_scripted: float = 0.75         # within pool play: scripted vs snapshot
    pool_every: int = 5              # snapshot cadence (iterations)
    pool_size: int = 8
    seed: int = 0
    log_every: int = 1


@dataclass
class Episode:
    obs: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)      # shaped, scaled
    raw_return: float = 0.0                                  # true utility

    def __len__(self):
        return len(self.actions)


class SelfPlayTrainer:
    def __init__(self, theory: TheorySpec, cfg: Optional[PPOConfig] = None,
                 env: Optional[StrategicEnduranceEnv] = None):
        self.theory = theory
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.env = env or StrategicEnduranceEnv(seed=self.cfg.seed)
        ed = theory.extra_feature_dim
        self.nets: Dict[str, SEEPolicy] = {
            p: SEEPolicy(OBS_DIM, N_ACTIONS, ed) for p in PLAYERS}
        self.opts = {p: torch.optim.Adam(self.nets[p].parameters(),
                                         lr=self.cfg.lr) for p in PLAYERS}
        self.pool: List[Dict[str, dict]] = []       # frozen snapshots
        self.iteration = 0
        self.total_steps = 0
        self.history: List[dict] = []

    # ---------------- rollout collection ---------------------------- #
    def _theory_obs(self, pid: str, obs: np.ndarray, public: dict
                    ) -> np.ndarray:
        if self.theory.extra_feature_dim == 0:
            return obs
        extra = np.asarray(
            self.theory.extra_features(pid, obs, public), dtype=np.float32)
        assert extra.shape == (self.theory.extra_feature_dim,), \
            f"extra_features must return shape ({self.theory.extra_feature_dim},)"
        extra = np.nan_to_num(extra, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.concatenate([obs, extra])

    def _frozen_opponent(self):
        """Sample a frozen opponent (scripted or snapshot) for one seat."""
        if not self.pool or self.rng.random() < self.cfg.p_scripted:
            name = self.rng.choice(list(BASELINES))
            return ("scripted", name)
        idx = int(self.rng.integers(len(self.pool)))
        return ("snapshot", idx)

    def collect(self, n_steps: int):
        """Collect ~n_steps environment steps of episodes."""
        batches: Dict[str, List[Episode]] = {p: [] for p in PLAYERS}
        steps = 0
        stats = {"len": [], "raw": {p: [] for p in PLAYERS},
                 "exit_self": 0, "n_eps": 0}
        while steps < n_steps:
            self_play = self.rng.random() < self.cfg.p_self
            learn_seats = set(PLAYERS)
            frozen_kind = frozen_net = frozen_agent = frozen_seat = None
            if not self_play:
                frozen_seat = self.rng.choice(PLAYERS)
                learn_seats = {p for p in PLAYERS if p != frozen_seat}
                kind, ref = self._frozen_opponent()
                frozen_kind = kind
                if kind == "scripted":
                    frozen_agent = BASELINES[ref]()
                    frozen_agent.reset(frozen_seat,
                                       seed=int(self.rng.integers(2**31)))
                else:
                    frozen_net = SEEPolicy(OBS_DIM, N_ACTIONS,
                                           self.theory.extra_feature_dim)
                    frozen_net.load_state_dict(self.pool[ref][frozen_seat])
                    frozen_net.eval()

            eps = {p: Episode() for p in PLAYERS}
            obs, _ = self.env.reset(seed=int(self.rng.integers(2**31)))
            self.theory.on_episode_start(IRAN)
            self.theory.on_episode_start(US)
            hidden = {p: None for p in PLAYERS}
            h_frozen = None
            done = False
            while not done:
                public = self.env.public_state()
                actions = {}
                step_data = {}
                for pid in PLAYERS:
                    mask = self.env.get_legal_mask(pid)
                    x = self._theory_obs(pid, obs[pid], public)
                    if pid in learn_seats:
                        a, lp, v, hidden[pid] = self.nets[pid].step(
                            torch.as_tensor(x), torch.as_tensor(mask),
                            hidden[pid], sample=True)
                        step_data[pid] = (x, mask, a, lp, v)
                    elif frozen_kind == "scripted":
                        a = frozen_agent.act(obs[pid], mask, public)
                    else:
                        a, _, _, h_frozen = frozen_net.step(
                            torch.as_tensor(x), torch.as_tensor(mask),
                            h_frozen, sample=True)
                    actions[pid] = a
                next_obs, rew, term, trunc, info = self.env.step(actions)
                done = term[IRAN] or trunc[IRAN]
                next_public = self.env.public_state()
                for pid in learn_seats:
                    x, mask, a, lp, v = step_data[pid]
                    shaped = rew[pid] + self.theory.shaping(
                        pid, obs[pid], a, rew[pid], next_obs[pid],
                        next_public, done)
                    ep = eps[pid]
                    ep.obs.append(x)
                    ep.masks.append(mask)
                    ep.actions.append(a)
                    ep.logps.append(lp)
                    ep.values.append(v)
                    ep.rewards.append(shaped * self.cfg.reward_scale)
                    ep.raw_return += rew[pid]
                obs = next_obs
                steps += 1
            for pid in learn_seats:
                if len(eps[pid]):
                    batches[pid].append(eps[pid])
                    stats["raw"][pid].append(eps[pid].raw_return)
            stats["len"].append(self.env.t)
            stats["n_eps"] += 1
        self.total_steps += steps
        return batches, stats

    # ---------------- PPO update ------------------------------------- #
    def _gae(self, ep: Episode):
        cfg = self.cfg
        T = len(ep)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            next_v = ep.values[t + 1] if t + 1 < T else 0.0   # terminal
            delta = ep.rewards[t] + cfg.gamma * next_v - ep.values[t]
            last = delta + cfg.gamma * cfg.gae_lambda * last
            adv[t] = last
        ret = adv + np.asarray(ep.values, dtype=np.float32)
        return adv, ret

    def update(self, batches: Dict[str, List[Episode]]):
        cfg = self.cfg
        losses = {}
        for pid in PLAYERS:
            eps = batches[pid]
            if not eps:
                continue
            data = []
            for ep in eps:
                adv, ret = self._gae(ep)
                data.append((ep, adv, ret))
            all_adv = np.concatenate([d[1] for d in data])
            mu, sd = all_adv.mean(), all_adv.std() + 1e-8
            net, opt = self.nets[pid], self.opts[pid]
            pl = vl = el = nb = 0.0
            for _ in range(cfg.epochs):
                order = self.rng.permutation(len(data))
                for start in range(0, len(order), cfg.minibatch_episodes):
                    mb = [data[i] for i in
                          order[start:start + cfg.minibatch_episodes]]
                    loss, p, v, e = self._ppo_loss(net, mb, mu, sd)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(),
                                             cfg.max_grad_norm)
                    opt.step()
                    pl, vl, el, nb = pl + p, vl + v, el + e, nb + 1
            losses[pid] = {"pi": pl / max(nb, 1), "v": vl / max(nb, 1),
                           "ent": el / max(nb, 1)}
        return losses

    def _ppo_loss(self, net, mb, adv_mu, adv_sd):
        cfg = self.cfg
        Tm = max(len(d[0]) for d in mb)
        B = len(mb)
        in_dim = OBS_DIM + self.theory.extra_feature_dim
        X = torch.zeros(B, Tm, in_dim)
        M = torch.zeros(B, Tm, N_ACTIONS, dtype=torch.bool)
        A = torch.zeros(B, Tm, dtype=torch.long)
        LP = torch.zeros(B, Tm)
        ADV = torch.zeros(B, Tm)
        RET = torch.zeros(B, Tm)
        PAD = torch.zeros(B, Tm, dtype=torch.bool)
        for b, (ep, adv, ret) in enumerate(mb):
            T = len(ep)
            X[b, :T] = torch.as_tensor(np.asarray(ep.obs))
            M[b, :T] = torch.as_tensor(np.asarray(ep.masks))
            A[b, :T] = torch.as_tensor(ep.actions)
            LP[b, :T] = torch.as_tensor(ep.logps)
            ADV[b, :T] = torch.as_tensor((adv - adv_mu) / adv_sd)
            RET[b, :T] = torch.as_tensor(ret)
            PAD[b, :T] = True
        logits, values, _ = net(X)
        logits = masked_logits(logits, M)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(A)
        ratio = torch.exp(logp - LP)
        s1 = ratio * ADV
        s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADV
        pad = PAD.float()
        n = pad.sum()
        pi_loss = -(torch.min(s1, s2) * pad).sum() / n
        v_loss = (((values - RET) ** 2) * pad).sum() / n
        ent = (dist.entropy() * pad).sum() / n
        loss = (pi_loss + cfg.value_coef * v_loss
                - cfg.entropy_coef * ent)
        return loss, pi_loss.item(), v_loss.item(), ent.item()

    # ---------------- main loop -------------------------------------- #
    def train(self, callback=None):
        cfg = self.cfg
        t0 = time.time()
        while self.total_steps < cfg.total_steps:
            self.iteration += 1
            batches, stats = self.collect(cfg.steps_per_iter)
            losses = self.update(batches)
            if self.iteration % cfg.pool_every == 0:
                self.pool.append({p: copy.deepcopy(self.nets[p].state_dict())
                                  for p in PLAYERS})
                if len(self.pool) > cfg.pool_size:
                    self.pool.pop(0)
            row = {
                "iter": self.iteration,
                "steps": self.total_steps,
                "ep_len": float(np.mean(stats["len"])),
                "raw_I": float(np.mean(stats["raw"][IRAN]))
                if stats["raw"][IRAN] else float("nan"),
                "raw_U": float(np.mean(stats["raw"][US]))
                if stats["raw"][US] else float("nan"),
                "time": time.time() - t0,
            }
            self.history.append(row)
            if self.iteration % cfg.log_every == 0:
                print(f"[iter {row['iter']:3d}] steps={row['steps']:>7d} "
                      f"len={row['ep_len']:5.1f} "
                      f"u_I={row['raw_I']:7.1f} u_U={row['raw_U']:7.1f} "
                      f"({row['time']:.0f}s)", flush=True)
            if callback:
                callback(self)
        return self.history

    # ---------------- persistence ------------------------------------ #
    def load_nets(self, path: str):
        """Warm-start from a previous checkpoint (nets only; optimizer and
        opponent pool restart). The loaded self is added to the pool so
        training immediately plays against its past self."""
        import torch as _t
        ck = _t.load(path, map_location="cpu", weights_only=True)
        assert ck.get("extra_dim", 0) == self.theory.extra_feature_dim, \
            "resume: theory extra_feature_dim changed"
        for pid in PLAYERS:
            self.nets[pid].load_state_dict(ck["nets"][pid])
        self.pool.append({p: copy.deepcopy(self.nets[p].state_dict())
                          for p in PLAYERS})
        prev = ck.get("train_meta", {}).get("total_steps", 0)
        print(f"warm-started from {path} (had {prev} steps)")

    def save_checkpoint(self, path: str, theory_source: str,
                        team: str = "anonymous"):
        torch.save({
            "version": 1,
            "team": team,
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "extra_dim": self.theory.extra_feature_dim,
            "theory_source": theory_source,
            "nets": {p: self.nets[p].state_dict() for p in PLAYERS},
            "train_meta": {
                "total_steps": self.total_steps,
                "iterations": self.iteration,
                "ppo": vars(self.cfg),
                "history_tail": self.history[-5:],
            },
        }, path)
        print(f"checkpoint saved -> {path}")
