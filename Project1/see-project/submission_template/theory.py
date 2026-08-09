"""Team theory v2.1 — belief filtering + potential-based shaping.

Direct implementation of the team's Section 4 analysis:
  * 4.3/4.4 (beliefs & costly signaling): particle filter over the opp's
    structural type (rho_bar, m_hat). Each particle simulates the opp's
    hidden endurance with the EXACT public cost law (g=(c+a+r)/m, ramp,
    K-stock), so "endurance spent = effective cost borne". Weights are
    updated by Bayes from the opp's ACTIVE signal choices via an
    ordered-logit model consistent with single-crossing:
        P(sigma <= k | s) = sigmoid(lam * (tau_k - s))   (decreasing in s)
    so stronger types put more mass on high signals.
    Refinement from 4.3/4.5: a trapped opponent (low X_j) continues even
    when weak, and cheap early signals (ramp) barely separate types, so
    each likelihood is raised to an *informativeness* power
    info = f(X_j) * ramp (weighted Bayes; bluffs/trapped play move the
    belief less).
  * 4.1 (war of attrition): the policy receives the terms of the
    continue-vs-exit inequality V^C >= X explicitly: X_i, X_j, B_i,
    effective (not raw) flow costs g_i, g_j, the exhaustion race and
    P(opp exits soon).
  * 4.2 (effective cost): features use g = raw/m, never raw pressure.
  * 4.4 (signal affordability): own phi denominator.
  * 4.5 (credibility vs flexibility): shaping potential
    Phi = p_win*B + (1-p_win)*X + 0.25*X. F = gamma*Phi(s') - Phi(s),
    Phi(terminal)=0: telescoping, optimal policy for raw utility unchanged.
Everything is deterministic given (obs, public_state).

v2.1 fixes: (1) broadcast crash in the likelihood (s (16,) vs tau (4,));
(2) ordered-logit orientation (was increasing in s, producing negative
middle-level probabilities that got clipped, so only extreme signals
updated the belief).
"""
import numpy as np

from see.config import canonical_config, IRAN, US
from see.training.theory_api import TheorySpec

# obs indices, for readability
T_FRAC, RHO, M_HAT, E_HAT, K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = range(9)

GAMMA = 0.995                      # trainer discount (DESIGN.md Sec. 9)
_CFG = canonical_config()
_RAMP_T = _CFG.ramp_T
_T_BAR = _CFG.T_bar

# ordered-logit signaling model (likelihood of the Bayes update, 4.4)
_TAU = np.array([0.30, 0.45, 0.60, 0.75])
_LAM = 6.0
# particle grids, weighted by the opponent's actual prior
_RHO_G = np.array([0.30, 0.50, 0.70, 0.90])
_MHAT_G = np.array([0.20, 0.45, 0.70, 0.90])
_SOON = 8.0                        # stages counted as "exits soon" (4.1)


def _other(pid):
    return US if pid == IRAN else IRAN


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class MyTheory(TheorySpec):
    name = "endurance-belief-v2.1"
    extra_feature_dim = 12

    def __init__(self):
        super().__init__()
        self._grids = {}
        self._cache = {}

    # ---------------- prior-weighted particle grid ------------------ #
    def _ensure_grid(self, opp_id):
        if opp_id in self._grids:
            return self._grids[opp_id]
        p = _CFG.players[opp_id]
        rho = np.repeat(_RHO_G, 4)
        mhat = np.tile(_MHAT_G, 4)
        lr = (p.rho_a - 1) * np.log(np.clip(rho, 1e-6, 1 - 1e-6)) + \
             (p.rho_b - 1) * np.log(np.clip(1 - rho, 1e-6, 1 - 1e-6))
        lm = (p.m_a - 1) * np.log(np.clip(mhat, 1e-6, 1 - 1e-6)) + \
             (p.m_b - 1) * np.log(np.clip(1 - mhat, 1e-6, 1 - 1e-6))
        w = np.exp(lr + lm - np.max(lr + lm)); w /= w.sum()
        self._grids[opp_id] = dict(
            rho=rho, mhat=mhat, w0=w,
            e0=100.0 * (0.50 + 0.35 * rho + 0.15 * mhat),
            mbar=p.m_lo + (p.m_hi - p.m_lo) * mhat,
            mean_rho=p.rho_a / (p.rho_a + p.rho_b))
        return self._grids[opp_id]

    def on_episode_start(self, player_id):
        self._cache.pop(player_id, None)

    # ---------------- weighted-Bayes endurance filter (4.3/4.4) ----- #
    def _belief(self, player_id, obs, public_state):
        opp_id = _other(player_id)
        g = self._ensure_grid(opp_id)
        op = _CFG.players[opp_id]
        rho, mhat, e0, mbar = g["rho"], g["mhat"], g["e0"], g["mbar"]
        w = g["w0"].copy()
        e = e0.copy()
        last_self = last_opp = 0.0
        K_opp = 0.0

        for row in public_state.get("history", []):
            t = row.get("t", 0)
            ramp = min(1.0, (t + 1) / _RAMP_T)
            # exact public cost law, per particle (DESIGN.md Sec. 5)
            raw = (op.c0 + op.kappa * last_self) + op.alpha * K_opp + \
                  op.eta * last_self * last_opp
            m = mbar * (0.35 + 0.65 * np.clip(e / e0, 0.0, 1.0))
            e = e - ramp * raw / m

            act = row.get("action", {}).get(opp_id, 6)
            if 0 <= act <= 4:                      # an informative choice
                lvl = int(round(row.get("sigma", {}).get(opp_id, 0.0) * 4))
                # strength: type + current endurance (single-crossing)
                s = 0.5 * rho + 0.3 * np.clip(e / e0, 0.0, 1.0) + 0.2 * mhat
                # P(sigma <= k | s), decreasing in s  ->  (16, 4)
                cdf = _sig(_LAM * (_TAU[None, :] - s[:, None]))
                P = np.empty((s.size, 5))
                P[:, 0] = cdf[:, 0]
                P[:, 1:4] = cdf[:, 1:] - cdf[:, :-1]
                P[:, 4] = 1.0 - cdf[:, 3]
                P = np.clip(P[:, lvl], 1e-4, 1.0)
                # 4.3/4.5: trapped opps continue even when weak and cheap
                # early signals barely separate -> damp the update
                X_j = op.x0 - 5.0 * K_opp - 8.0 * last_self + \
                      20.0 * row.get("mediation", 0)
                info = np.clip((X_j - (op.x0 - 20.0)) / 20.0, 0.05, 1.0)
                info *= ramp
                w *= P ** info
                w /= w.sum()

            last_self = row.get("sigma", {}).get(player_id, last_self)
            s_opp = row.get("sigma", {}).get(opp_id, last_opp)
            K_opp = 0.75 * K_opp + s_opp
            last_opp = s_opp
            if row.get("exit", {}).get(opp_id) or \
                    row.get("exit", {}).get(player_id):
                break

        eh = np.clip(e / e0, 0.0, 1.0)
        return dict(w=w, e=e, e0=e0, mbar=mbar, eh=eh,
                    ej_hat=float(w @ eh),
                    ej_std=float(np.sqrt(w @ (eh - (w @ eh)) ** 2)),
                    e0_hat=float(w @ e0), mbar_hat=float(w @ mbar))

    # ---------------- exact own-side analytics (4.1/4.2/4.4) -------- #
    def _own(self, player_id, obs):
        pp = _CFG.players[player_id]
        ramp = min(1.0, (obs[T_FRAC] * _T_BAR + 1) / _RAMP_T)
        K_i = obs[K_OWN] * 4.0
        e_hat = max(obs[E_HAT], 0.0)
        mbar_i = pp.m_lo + (pp.m_hi - pp.m_lo) * obs[M_HAT]
        m_i = mbar_i * (0.35 + 0.65 * e_hat)
        raw = (pp.c0 + pp.kappa * obs[SIG_OPP]) + pp.alpha * K_i + \
              pp.eta * obs[SIG_OWN] * obs[SIG_OPP]
        g_i = ramp * raw / m_i                              # 4.2: raw/m
        X_i = pp.x0 - 5.0 * K_i - 8.0 * obs[SIG_OPP] + 20.0 * obs[MED]
        B_i = pp.b0 * (0.50 + 0.30 * e_hat + 0.20 * obs[RHO])
        denom = (0.4 + 0.6 * obs[RHO]) * (0.4 + 0.6 * min(e_hat, 1.0)) * \
                (0.5 + 0.5 * obs[M_HAT])                    # 4.4 afford.
        e0_i = 100.0 * (0.50 + 0.35 * obs[RHO] + 0.15 * obs[M_HAT])
        return dict(g_i=g_i, X_i=X_i, B_i=B_i, denom=denom, e0_i=e0_i,
                    ramp=ramp)

    # ---------------- belief features for the policy ---------------- #
    def extra_features(self, player_id, obs, public_state):
        opp_id = _other(player_id)
        op = _CFG.players[opp_id]
        bel = self._belief(player_id, obs, public_state)
        own = self._own(player_id, obs)

        # opp effective cost & horizons per particle (4.1 race, 4.2)
        raw_j = (op.c0 + op.kappa * obs[SIG_OWN]) + \
                op.alpha * obs[K_OPP] * 4.0 + \
                op.eta * obs[SIG_OWN] * obs[SIG_OPP]
        m_j = bel["mbar"] * (0.35 + 0.65 * bel["eh"])
        g_j = own["ramp"] * raw_j / np.maximum(m_j, 1e-3)
        e_j_abs = bel["eh"] * bel["e0"]
        hor_j = np.clip(e_j_abs / np.maximum(g_j, 0.5), 0.0, 40.0)
        g_j_hat = float(bel["w"] @ g_j)
        p_soon = float(bel["w"] @ (hor_j <= _SOON))          # mu(tau_j small)
        opp_h = float(bel["w"] @ hor_j)
        own_h = float(np.clip(obs[E_HAT] * own["e0_i"] /
                              max(own["g_i"], 0.5), 0.0, 40.0))

        X_j = op.x0 - 5.0 * obs[K_OPP] * 4.0 - 8.0 * obs[SIG_OWN] + \
              20.0 * obs[MED]                                # 4.5: trapped?
        calm = (obs[SIG_OWN] + obs[SIG_OPP]) <= _CFG.calm_thresh
        med_out = (1.0 - _CFG.p_close) if obs[MED] > 0.5 else \
                  (_CFG.p_open_calm if calm else _CFG.p_open_hot)

        feats = np.array([
            bel["ej_hat"],                 # 0 opp endurance (posterior mean)
            bel["ej_std"],                 # 1 residual uncertainty mu
            obs[E_HAT] - bel["ej_hat"],    # 2 relative strength
            own["X_i"] / 46.0,             # 3 own exit value (4.1/4.5)
            X_j / 46.0,                    # 4 opp exit value / entrapment
            own["B_i"] / 175.0,            # 5 prize for outlasting
            min(own["g_i"] / 8.0, 2.0),    # 6 own EFFECTIVE cost (4.2)
            min(g_j_hat / 8.0, 2.0),       # 7 opp effective cost (belief)
            (own_h - opp_h) / 40.0,        # 8 exhaustion race (4.1)
            p_soon,                        # 9 P(opp folds soon)
            own["denom"],                  # 10 signal affordability (4.4)
            med_out,                       # 11 face-saving exit outlook
        ], dtype=np.float32)

        self._cache[player_id] = bel
        return feats

    # ---------------- potential-based shaping (4.1/4.5) ------------- #
    def _potential(self, player_id, obs, bel):
        own = self._own(player_id, obs)
        opp_id = _other(player_id)
        op = _CFG.players[opp_id]
        raw_j = (op.c0 + op.kappa * obs[SIG_OWN]) + \
                op.alpha * obs[K_OPP] * 4.0 + \
                op.eta * obs[SIG_OWN] * obs[SIG_OPP]
        m_j = bel["mbar_hat"] * (0.35 + 0.65 * bel["ej_hat"])
        g_j = own["ramp"] * raw_j / max(m_j, 1e-3)
        opp_h = np.clip(bel["ej_hat"] * bel["e0_hat"] / max(g_j, 0.5), 0, 40)
        own_h = np.clip(obs[E_HAT] * own["e0_i"] / max(own["g_i"], 0.5), 0, 40)
        mean_rho_opp = self._ensure_grid(opp_id)["mean_rho"]
        # 4.1: subjective probability of winning the attrition race
        p_win = _sig(2.0 * (obs[E_HAT] - bel["ej_hat"]) +
                     0.06 * (own_h - opp_h) +
                     0.6 * (obs[RHO] - mean_rho_opp))
        # V^C approximation + 0.25*X: option value of keeping exit open
        return (p_win * own["B_i"] + (1.0 - p_win) * own["X_i"] +
                0.25 * own["X_i"]) / 20.0

    def shaping(self, player_id, obs, action, env_reward, next_obs,
                next_public, terminated):
        bel_cur = self._cache.get(player_id)
        if bel_cur is None:
            bel_cur = self._belief(player_id, obs, next_public)
        phi_cur = self._potential(player_id, obs, bel_cur)
        if terminated:
            phi_next = 0.0                    # telescoping exact
        else:
            bel_next = self._belief(player_id, next_obs, next_public)
            phi_next = self._potential(player_id, next_obs, bel_next)
        return float(GAMMA * phi_next - phi_cur)


# ------------------------------------------------------------------ #
# quick self-test: python submission_template/theory.py
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    th = MyTheory()
    obs = np.full(16, 0.3, dtype=np.float32)
    obs[RHO], obs[M_HAT], obs[E_HAT] = 0.7, 0.5, 0.9
    hist = [dict(t=t, sigma={IRAN: 0.5, US: 0.75},
                 action={IRAN: 2, US: 3},
                 exit={IRAN: False, US: False}, mediation=0)
            for t in range(6)]
    pub = dict(t=6, T_bar=_T_BAR, K={IRAN: 1.0, US: 1.0},
               last_sigma={IRAN: 0.5, US: 0.75}, mediation=0,
               active={IRAN: True, US: True}, history=hist)
    for pid in (IRAN, US):
        th.on_episode_start(pid)
        f = th.extra_features(pid, obs, pub)
        assert f.shape == (th.extra_feature_dim,), f.shape
        assert np.isfinite(f).all()
        r = th.shaping(pid, obs, 2, -1.0, obs, pub, False)
        assert np.isfinite(r)
    print("self-test OK: features + shaping run clean for both roles")