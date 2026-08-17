"""YOUR THEORY GOES HERE.

This file is your team's core intellectual contribution: it translates your
Section 4 theoretical analysis (see the project manual) into (1) explicit
belief features and (2) a training objective. WHAT to compute in each hook
is exactly the content of your Section 4 work — this template only shows
you WHERE it plugs in. Edit it, then train:

    python scripts/train.py --theory submission_template/theory.py \
        --team "your-team-name" --out submission.pt

The environment itself (see/config.py, see/env.py) is FIXED — the
tournament runs the canonical game for everyone. Everything you are allowed
to condition on is in your own observation vector and the public state:

obs indices (see see/env.py for the full layout):
    0 t/T_bar | 1 own rho | 2 own m_hat | 3 own e_hat | 4 own K/4
    5 opp K/4 | 6 own last sigma | 7 opp last sigma | 8 mediation
    9 self active | 10 opp active | 11 opp high-signal freq
    12 opp mean sigma | 13 own mean sigma | 14 own phi spend/20
    15 own G/e0

public_state dict:
    t, T_bar, K, last_sigma, mediation, active,
    history = [{t, sigma:{I,U}, action:{I,U}, exit:{I,U}, mediation}, ...]

You may use anything in `obs` and `public_state` (your own private state +
the public history). You have NO access to the opponent's type or
endurance. See `see/training/theory_api.py` for the full contract.

This template ships as a NULL theory: no extra features, no shaping — it
trains against the plain environment reward and is a valid (if weak)
submission as-is. Replace the two methods below with your own analysis.
"""
import numpy as np

from see.training.theory_api import TheorySpec

# obs indices, for readability in your own code
T_FRAC, RHO, M_HAT, E_HAT, K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = range(9)


class MyTheory(TheorySpec):
    """Replace with your own analysis (rename freely)."""

    name = "bayesian-endurance-theory"

    # Number of extra belief features you append each stage. Set this to
    # the length of the vector extra_features() returns. Leave 0 for the
    # null theory below.
    extra_feature_dim = 16

    def on_episode_start(self, player_id):
        """Called once at the start of every episode. Reset any per-episode
        state here (e.g. a belief filter you carry across stages)."""
        # The filter below is deliberately recomputed from public history.
        # This keeps it deterministic and avoids cross-talk because the
        # trainer uses one TheorySpec instance for both players.
        pass

    # ------------------------------------------------------------------ #
    # (1) BELIEF FEATURES — appended to the policy input every stage.     #
    #     Called during training AND evaluation, so it must be            #
    #     deterministic given (obs, public_state).                        #
    #                                                                     #
    #     Return a float32 array of length `extra_feature_dim`. WHAT to   #
    #     put here is your Section 4.3 analysis (what can you infer about #
    #     the opponent from your own state and the public history?). This #
    #     template returns nothing.                                       #
    # ------------------------------------------------------------------ #
    def extra_features(self, player_id, obs, public_state):
        opponent = "U" if player_id == "I" else "I"

        # A deterministic quadrature approximation to the posterior over
        # theta_j=(rho_j,m_hat_j).  The prior is the published asymmetric
        # beta prior.  Signal likelihoods implement single crossing: costly
        # signals are more likely for types for which they are affordable.
        q = np.linspace(0.05, 0.95, 9, dtype=np.float64)
        rho, m_hat = np.meshgrid(q, q, indexing="ij")
        rho = rho.ravel()
        m_hat = m_hat.ravel()
        if opponent == "I":
            rho_a, rho_b, m_a, m_b = 5.0, 2.0, 2.0, 2.0
            m_lo, m_hi = 0.8, 1.7
            c0, kappa, alpha, x0 = 1.4, 3.0, 0.8, 26.0
        else:
            rho_a, rho_b, m_a, m_b = 3.0, 3.0, 3.0, 2.0
            m_lo, m_hi = 0.9, 1.8
            c0, kappa, alpha, x0 = 1.8, 2.5, 1.3, 28.0

        weights = (rho ** (rho_a - 1.0) * (1.0 - rho) ** (rho_b - 1.0)
                   * m_hat ** (m_a - 1.0)
                   * (1.0 - m_hat) ** (m_b - 1.0))
        weights /= weights.sum()
        m_bar = m_lo + (m_hi - m_lo) * m_hat
        e0 = 100.0 * (0.50 + 0.35 * rho + 0.15 * m_hat)
        endurance = e0.copy()
        k_opp = 0.0
        history = public_state.get("history", ())

        for h in history:
            sig = h.get("sigma", {})
            sigma_opp = float(sig.get(opponent, 0.0))
            sigma_self = float(sig.get(player_id, 0.0))
            active = h.get("active", ())
            action = int(h.get("action", {}).get(opponent, 6))
            e_hat_particles = np.clip(endurance / e0, 0.0, 1.2)

            # Active choices are informative, whereas a forced HOLD is not.
            # A small contamination floor acknowledges pooling/bluffing and
            # prevents a single off-path action from collapsing the filter.
            if opponent in active and action < 5:
                phi = (1.2 * sigma_opp ** 2
                       / (0.4 + 0.6 * rho)
                       / (0.4 + 0.6 * np.minimum(e_hat_particles, 1.0))
                       / (0.5 + 0.5 * m_hat))
                strength = (0.45 * rho + 0.25 * m_hat
                            + 0.30 * np.minimum(e_hat_particles, 1.0))
                target = np.clip(0.05 + 0.62 * strength
                                 + 0.24 * sigma_self - 0.10 * k_opp / 4.0,
                                 0.0, 1.0)
                likelihood = np.exp(-0.5 * ((sigma_opp - target) / 0.28) ** 2
                                    - 0.10 * phi)
                weights *= 0.18 + 0.82 * likelihood
                weights /= max(weights.sum(), 1e-300)

            t_stage = int(h.get("t", 0))
            capacity = m_bar * (0.35 + 0.65 * e_hat_particles)
            raw_cost = (c0 + kappa * sigma_self + alpha * k_opp
                        + 2.5 * sigma_opp * sigma_self)
            flow_cost = min(1.0, (t_stage + 1.0) / 4.0) * raw_cost / capacity
            endurance -= flow_cost
            k_opp = 0.75 * k_opp + sigma_opp

        e_hat_particles = np.clip(endurance / e0, 0.0, 1.2)

        # Reaching another decision node is itself censored evidence that
        # the opponent has endurance left.  The logistic width reflects the
        # accumulated N(0,.5^2) endurance shocks without pretending that the
        # approximate filter observes those shocks.
        if opponent in public_state.get("active", ()) and history:
            noise_width = 0.55 * np.sqrt(len(history)) + 0.35
            survival = 1.0 / (1.0 + np.exp(
                -np.clip(endurance / noise_width, -40.0, 40.0)))
            weights *= 0.03 + 0.97 * survival
            weights /= max(weights.sum(), 1e-300)

        post_rho = float(np.sum(weights * rho))
        post_m = float(np.sum(weights * m_hat))
        post_e = float(np.sum(weights * e_hat_particles))
        post_e_sd = float(np.sqrt(np.sum(
            weights * (e_hat_particles - post_e) ** 2)))
        weak_prob = float(np.sum(weights[e_hat_particles < 0.25]))
        strong_prob = float(np.sum(weights[
            (rho > 0.65) & (m_hat > 0.50) & (e_hat_particles > 0.50)]))

        t_now = int(public_state.get("t", len(history)))
        sigma_opp_now = float(public_state.get(
            "last_sigma", {}).get(opponent, obs[SIG_OPP]))
        sigma_self_now = float(public_state.get(
            "last_sigma", {}).get(player_id, obs[SIG_OWN]))
        k_opp_now = float(public_state.get(
            "K", {}).get(opponent, 4.0 * obs[K_OPP]))
        capacity = m_bar * (0.35 + 0.65 * e_hat_particles)
        next_raw = (c0 + kappa * sigma_self_now + alpha * k_opp_now
                    + 2.5 * sigma_opp_now * sigma_self_now)
        next_g = min(1.0, (t_now + 1.0) / 4.0) * next_raw / capacity
        expected_g = float(np.sum(weights * next_g))
        credible_phi = (1.2 * 0.75 ** 2
                        / (0.4 + 0.6 * rho)
                        / (0.4 + 0.6 * np.minimum(e_hat_particles, 1.0))
                        / (0.5 + 0.5 * m_hat))
        expected_phi = float(np.sum(weights * credible_phi))

        opp_b = 175.0 * (0.50 + 0.30 * post_e + 0.20 * post_rho)
        mediation = float(obs[MED])
        opp_x = x0 - 5.0 * k_opp_now - 8.0 * sigma_self_now + 20.0 * mediation
        opp_wedge = (opp_b - opp_x) / 175.0

        own_x0 = 26.0 if player_id == "I" else 28.0
        if player_id == "I":
            own_m_lo, own_m_hi = 0.8, 1.7
            own_c0, own_kappa, own_alpha = 1.4, 3.0, 0.8
        else:
            own_m_lo, own_m_hi = 0.9, 1.8
            own_c0, own_kappa, own_alpha = 1.8, 2.5, 1.3
        own_b = 175.0 * (0.50 + 0.30 * float(obs[E_HAT])
                         + 0.20 * float(obs[RHO]))
        own_x = (own_x0 - 20.0 * float(obs[K_OWN])
                 - 8.0 * float(obs[SIG_OPP]) + 20.0 * mediation)
        own_wedge = (own_b - own_x) / 175.0
        own_m_bar = (own_m_lo + (own_m_hi - own_m_lo)
                     * float(obs[M_HAT]))
        own_capacity = own_m_bar * (
            0.35 + 0.65 * max(float(obs[E_HAT]), 0.0))
        own_raw = (own_c0 + own_kappa * float(obs[SIG_OPP])
                   + own_alpha * 4.0 * float(obs[K_OWN])
                   + 2.5 * float(obs[SIG_OWN]) * float(obs[SIG_OPP]))
        own_g = min(1.0, (t_now + 1.0) / 4.0) * own_raw / own_capacity
        continue_margin = (weak_prob * own_wedge - own_g / 20.0
                           + 0.10 * max(float(obs[E_HAT]) - post_e, 0.0))
        own_phi = (1.2 * 0.75 ** 2
                   / (0.4 + 0.6 * float(obs[RHO]))
                   / (0.4 + 0.6 * min(float(obs[E_HAT]), 1.0))
                   / (0.5 + 0.5 * float(obs[M_HAT])))

        active_opp_signals = [
            float(h.get("sigma", {}).get(opponent, 0.0))
            for h in history if opponent in h.get("active", ())
            and int(h.get("action", {}).get(opponent, 6)) < 5
        ]
        if active_opp_signals:
            recent = np.asarray(active_opp_signals[-6:], dtype=np.float64)
            recency_weights = np.arange(1.0, len(recent) + 1.0)
            recent_signal = float(np.dot(recent, recency_weights)
                                  / recency_weights.sum())
            if len(recent) > 1:
                split = len(recent) // 2
                signal_trend = float(np.mean(recent[split:])
                                     - np.mean(recent[:split]))
            else:
                signal_trend = 0.0
        else:
            recent_signal = 0.0
            signal_trend = 0.0

        feats = np.asarray([
            post_rho,
            post_m,
            np.clip(post_e / 1.2, 0.0, 1.0),
            np.clip(post_e_sd / 0.35, 0.0, 1.0),
            np.clip(weak_prob, 0.0, 1.0),
            np.clip(strong_prob, 0.0, 1.0),
            np.clip(expected_g / 12.0, 0.0, 1.0),
            np.clip(expected_phi / 5.0, 0.0, 1.0),
            np.clip(recent_signal, 0.0, 1.0),
            np.clip(signal_trend, -1.0, 1.0),
            np.clip(opp_wedge, -1.0, 1.0),
            np.clip(own_wedge, -1.0, 1.0),
            np.clip(float(obs[E_HAT]) - post_e, -1.0, 1.0),
            np.clip(continue_margin, -1.0, 1.0),
            np.clip(1.0 - own_phi / 5.0, 0.0, 1.0),
            np.clip((own_x + 20.0) / 70.0, 0.0, 1.0),
        ], dtype=np.float32)
        return feats

    # ------------------------------------------------------------------ #
    # (2) REWARD SHAPING — TRAINING ONLY. Added to the environment        #
    #     reward at each step. The leaderboard always scores the game's   #
    #     RAW utility u_i, so shaping only changes what your agent learns #
    #     to want, not how it is judged.                                  #
    #                                                                     #
    #     WHAT to reward is your Section 4 analysis. A safe form is       #
    #     potential-based shaping, F = gamma*Phi(s') - Phi(s), which      #
    #     telescopes over an episode and so provably leaves the optimal   #
    #     policy unchanged (Ng, Harada & Russell 1999); Phi is a function #
    #     YOU choose. A reward that does NOT telescope (e.g. a flat bonus #
    #     per stage survived) changes the objective and will train an     #
    #     agent that optimizes your bonus instead of the game.            #
    # ------------------------------------------------------------------ #
    def shaping(self, player_id, obs, action, env_reward, next_obs,
                next_public, terminated):
        # Phi values strategic option value: endurance and the better of
        # outlasting versus a politically feasible exit.  This is guidance,
        # not a new preference: gamma*Phi(s')-Phi(s), with Phi(terminal)=0,
        # telescopes under the trainer's gamma=0.995 and preserves the raw
        # game's optimal policy.
        def potential(o):
            rho = float(o[RHO])
            e_hat = float(o[E_HAT])
            k_own = float(o[K_OWN])
            mediation = float(o[MED])
            b_value = 175.0 * (0.50 + 0.30 * e_hat + 0.20 * rho)
            x0 = 26.0 if player_id == "I" else 28.0
            x_value = (x0 - 20.0 * k_own - 8.0 * float(o[SIG_OPP])
                       + 20.0 * mediation)
            p_outlast = np.clip(0.15 + 0.35 * rho + 0.30 * e_hat
                                + 0.20 * float(o[M_HAT]), 0.0, 1.0)
            strategic_value = p_outlast * b_value + (1.0 - p_outlast) * x_value
            return 0.12 * strategic_value + 4.0 * e_hat - 2.0 * k_own

        phi_now = potential(obs)
        phi_next = 0.0 if terminated else potential(next_obs)
        return float(0.995 * phi_next - phi_now)
