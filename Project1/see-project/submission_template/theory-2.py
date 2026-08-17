from sys import exception

import numpy as np

from see.training.theory_api import TheorySpec
from see.config import canonical_config, IRAN, US


# Observation indices from see/env.py
T_FRAC, RHO, M_HAT, E_HAT, K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = range(9)

GAMMA = 0.995
CFG = canonical_config()


class MyTheory(TheorySpec):
    """Simple theory features designed for robust baseline play."""

    name = "theory"
    extra_feature_dim = 9

    def on_episode_start(self, player_id):
        # No hidden memory is needed.
        # Everything is recomputed from obs + public history, which keeps the
        # theory deterministic during both training and evaluation.
        pass

    # ------------------------------------------------------------------ #
    # Basic game helpers
    # ------------------------------------------------------------------ #

    def _opponent_id(self, player_id):
        return US if player_id == IRAN else IRAN

    def _exit_value(self, pid, K, sigma_opp, M):
        """Exact public exit value X_i from the environment."""
        pp = CFG.players[pid]
        return (
            pp.x0
            - pp.lam * K
            - pp.h * sigma_opp
            + pp.w_med * M
        )

    def _own_flow_cost(self, player_id, obs):
        """Exact current-state flow cost g_i using the observed state."""
        pp = CFG.players[player_id]

        K = float(obs[K_OWN]) * 4.0
        e_hat = max(float(obs[E_HAT]), 0.0)

        m_bar = pp.m_lo + (pp.m_hi - pp.m_lo) * float(obs[M_HAT])
        m_t = m_bar * (
            pp.m_floor + (1.0 - pp.m_floor) * min(e_hat, 1.0)
        )

        t = int(round(float(obs[T_FRAC]) * CFG.T_bar))
        ramp = min(1.0, (t + 1) / CFG.ramp_T)

        # obs contains the CURRENT standing postures in the state.
        raw = (
            pp.c0
            + pp.kappa * float(obs[SIG_OPP])
            + pp.alpha * K
            + pp.eta * float(obs[SIG_OWN]) * float(obs[SIG_OPP])
        )

        return ramp * raw / m_t if m_t > 0.0 else 0.0

    # ------------------------------------------------------------------ #
    # Public-history statistics
    # ------------------------------------------------------------------ #

    def _history_statistics(self, player_id, public_state):
        opp = self._opponent_id(player_id)
        history = public_state.get("history", [])

        if not history:
            return {
                "initiative_mean": 0.0,
                "initiative_high": 0.0,
                "initiative_count": 0,
                "response_match": 0.0,
                "response_count": 0,
            }

        initiative_sum = 0.0
        initiative_high = 0
        initiative_count = 0

        response_matches = 0
        response_count = 0

        # This is the player's standing signal immediately BEFORE each stage.
        previous_own_sigma = 0.0

        for h in history:
            active = set(h.get("active", ()))
            opp_sigma = float(h["sigma"].get(opp, 0.0))

            if len(active) == 2:
                # Both players chose their new posture. This is the cleanest
                # public evidence about the opponent's own preference.
                initiative_sum += opp_sigma
                initiative_count += 1
                if opp_sigma >= 0.75:
                    initiative_high += 1

            elif len(active) == 1 and opp in active:
                # The opponent was responding to us. Compare its new signal
                # with our previous standing signal. Exact equality is natural
                # here because the signal ladder has only five values.
                response_count += 1
                if abs(opp_sigma - previous_own_sigma) < 1e-8:
                    response_matches += 1

            # The signal chosen in this stage becomes the standing signal for
            # the next stage.
            previous_own_sigma = float(h["sigma"].get(player_id, 0.0))

        return {
            "initiative_mean": (
                initiative_sum / initiative_count
                if initiative_count > 0 else 0.0
            ),
            "initiative_high": (
                initiative_high / initiative_count
                if initiative_count > 0 else 0.0
            ),
            "initiative_count": initiative_count,
            "response_match": (
                response_matches / response_count
                if response_count > 0 else 0.0
            ),
            "response_count": response_count,
        }

    def _estimated_type(self, player_id, public_state):
        pp = CFG.players[self._opponent_id(player_id)]
        stats = self._history_statistics(player_id, public_state)

        aggression = (
            0.6 * stats["initiative_high"]
            + 0.4 * stats["initiative_mean"]
        )

        stages = stats["initiative_count"]
        update_weight = min(stages / 10.0, 0.7)

        prior_rho = pp.rho_a / (pp.rho_a + pp.rho_b)

        est_rho = (
            (1.0 - update_weight) * prior_rho
            + update_weight * aggression
        )

        est_m = (
            (1.0 - update_weight) * 0.5
            + update_weight * aggression
        )

        return (
            float(np.clip(est_rho, 0.0, 1.0)),
            float(np.clip(est_m, 0.0, 1.0)),
        )

    def _p_outlasting(self, player_id, obs, public_state):
        # Our current endurance

        # e_hat is normalized by our own initial endurance.
        e_own_hat = max(float(obs[E_HAT]), 0.0)

        # Recover our actual initial endurance.
        pp_own = CFG.players[player_id]

        m_hat_own = float(np.clip(obs[M_HAT], 0.0, 1.0))
        rho_own = float(np.clip(obs[RHO], 0.0, 1.0))

        e0_own = pp_own.E0 * (
            pp_own.e0_base
            + pp_own.e0_rho * rho_own
            + pp_own.e0_m * m_hat_own
        )

        e_own = e_own_hat * e0_own

        # Estimate opponent endurance
        est_rho, est_m = self._estimated_type(
            player_id,
            public_state,
        )

        e_opp_hat = self._estimated_opponent_endurance(
            player_id,
            public_state,
            est_rho,
            est_m,
        )

        opp = self._opponent_id(player_id)
        pp_opp = CFG.players[opp]

        m_bar_opp = pp_opp.m_lo + (
            pp_opp.m_hi - pp_opp.m_lo
        ) * est_m

        e0_opp = pp_opp.E0 * (
            pp_opp.e0_base
            + pp_opp.e0_rho * est_rho
            + pp_opp.e0_m * est_m
        )

        e_opp = e_opp_hat * e0_opp

        # Estimate current endurance consumption
        g_own = self._own_flow_cost(
            player_id,
            obs,
        )

        g_opp = self._estimated_opponent_flow_cost(
            player_id,
            obs,
            est_m,
            e_opp_hat,
        )

        # Avoid division by zero when the player is currently paying
        # almost no flow cost.
        EPS = 1e-6

        g_own = max(g_own, EPS)
        g_opp = max(g_opp, EPS)

        # Estimated time until endurance exhaustion
        tau_own = e_own / g_own
        tau_opp = e_opp / g_opp

        # The game itself has a finite horizon.
        tau_own = min(tau_own, float(CFG.T_bar))
        tau_opp = min(tau_opp, float(CFG.T_bar))

        # Convert the time advantage into a probability
        TIME_SCALE = 4.0

        z = (tau_own - tau_opp) / TIME_SCALE
        z = float(np.clip(z, -8.0, 8.0))

        p_outlast = 1.0 / (1.0 + np.exp(-z))

        return float(p_outlast)

    # ------------------------------------------------------------------ #
    # Hidden-endurance estimate
    # ------------------------------------------------------------------ #

    def _estimated_opponent_endurance(
        self, player_id, public_state, est_rho, est_m
    ):
        opp = self._opponent_id(player_id)
        pp = CFG.players[opp]

        m_bar = pp.m_lo + (pp.m_hi - pp.m_lo) * est_m
        e0 = pp.E0 * (
            pp.e0_base
            + pp.e0_rho * est_rho
            + pp.e0_m * est_m
        )

        if e0 <= 0.0:
            return 0.0

        estimated_e = e0
        K_opp = 0.0

        for h in public_state.get("history", []):
            t = int(h.get("t", 0))
            sigma_opp = float(h["sigma"].get(opp, 0.0))
            sigma_own = float(h["sigma"].get(player_id, 0.0))

            e_hat = max(estimated_e / e0, 0.0)
            m_t = m_bar * (
                pp.m_floor
                + (1.0 - pp.m_floor) * min(e_hat, 1.0)
            )

            ramp = min(1.0, (t + 1) / CFG.ramp_T)

            raw = (
                pp.c0
                + pp.kappa * sigma_opp * 0.0
            )

            # For the opponent, material pressure comes from OUR standing
            # signal. Audience cost comes from the opponent's own K. Risk is
            # the product of both standing signals.
            raw = (
                pp.c0
                + pp.kappa * sigma_own
                + pp.alpha * K_opp
                + pp.eta * sigma_opp * sigma_own
            )

            if m_t > 0.0:
                estimated_e -= ramp * raw / m_t

            # The current action changes K only AFTER current-stage cost has
            # been charged, exactly as in see/env.py.
            K_opp = CFG.delta_K * K_opp + sigma_opp

        return float(np.clip(max(estimated_e, 0.0) / e0, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Strategic quantities
    # ------------------------------------------------------------------ #

    def _estimated_opponent_flow_cost(self, player_id, obs, est_m, est_e):
        """Estimate the opponent's current effective flow cost."""
        opp = self._opponent_id(player_id)
        pp = CFG.players[opp]

        K_opp = float(obs[K_OPP]) * 4.0
        sigma_opp = float(obs[SIG_OPP])
        sigma_own = float(obs[SIG_OWN])

        m_bar = pp.m_lo + (pp.m_hi - pp.m_lo) * est_m
        m_t = m_bar * (
            pp.m_floor + (1.0 - pp.m_floor) * min(est_e, 1.0)
        )

        t = int(round(float(obs[T_FRAC]) * CFG.T_bar))
        ramp = min(1.0, (t + 1) / CFG.ramp_T)

        raw = (
            pp.c0
            + pp.kappa * sigma_own
            + pp.alpha * K_opp
            + pp.eta * sigma_opp * sigma_own
        )

        return ramp * raw / m_t if m_t > 0.0 else 0.0

    def _continuation_margin(self, player_id, obs, est_e, public_state):
        """Approximate p(win)*(B-X)-g for the current player."""
        pp = CFG.players[player_id]

        e_hat = float(np.clip(obs[E_HAT], 0.0, 1.0))

        p_win = self._p_outlasting(player_id, obs, public_state)

        B = pp.b0 * (
            pp.b_base
            + pp.b_e * e_hat
            + pp.b_rho * float(obs[RHO])
        )

        X = self._exit_value(
            player_id,
            float(obs[K_OWN]) * 4.0,
            float(obs[SIG_OPP]),
            float(obs[MED]),
        )

        g = self._own_flow_cost(player_id, obs)

        return float(np.tanh((p_win * (B - X) - g) / 40.0))

    # ------------------------------------------------------------------ #
    # Belief/state features
    # ------------------------------------------------------------------ #

    def extra_features(self, player_id, obs, public_state):
        """Return 10 compact theory features."""
        opp = self._opponent_id(player_id)

        est_rho, est_m = self._estimated_type(
            player_id, public_state
        )
        est_e = self._estimated_opponent_endurance(
            player_id, public_state, est_rho, est_m
        )

        margin = self._continuation_margin(
            player_id, obs, est_e, public_state
        )

        # Feature: relative endurance. Positive means we probably have more.
        own_e = max(float(obs[E_HAT]), 0.0)
        relative_endurance = float(
            np.clip((own_e - est_e) / 1.0, -1.0, 1.0)
        )

        # Feature: estimated opponent current cost.
        opponent_cost = self._estimated_opponent_flow_cost(
            player_id, obs, est_m, est_e
        )

        # Exact public exit values.
        X_i = self._exit_value(
            player_id,
            float(obs[K_OWN]) * 4.0,
            float(obs[SIG_OPP]),
            float(obs[MED]),
        )
        X_j = self._exit_value(
            opp,
            float(obs[K_OPP]) * 4.0,
            float(obs[SIG_OWN]),
            float(obs[MED]),
        )

        g_i = self._own_flow_cost(player_id, obs)

        # Generic responsiveness statistic: how often the opponent mirrors
        # our previous standing signal on response turns. This becomes high
        # against Tit-for-Tat, but it is NOT hard-coded as "the opponent is
        # TFT" and therefore remains meaningful for other policies too.
        stats = self._history_statistics(player_id, public_state)
        response_score = stats["response_match"]

        # Expected mediation outlook from the public Markov rule.
        calm = (
            float(obs[SIG_OWN]) + float(obs[SIG_OPP])
            <= CFG.calm_thresh
        )
        if float(obs[MED]) > 0.5:
            mediation_outlook = 1.0 - CFG.p_close
        else:
            mediation_outlook = (
                CFG.p_open_calm if calm else CFG.p_open_hot
            )

        return np.array(
            [
                est_rho,                       # 0 opponent resolve
                est_m,                         # 1 opponent cost management
                est_e,                         # 2 opponent endurance
                margin,                         # 3 continuation margin
                relative_endurance,             # 4 own - opponent endurance
                min(opponent_cost / 8.0, 2.0), # 5 opponent flow cost response_score 6 response/TFT-likeness
                X_i / 48.0,                    # 7 own exit value
                X_j / 48.0,                    # 8 opponent exit value
                min(g_i / 8.0, 2.0),           # 9 own flow cost
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Reward shaping
    # ------------------------------------------------------------------ #

    def _current_public_state(self, next_public):
        history = next_public.get("history", [])

        current_public = dict(next_public)

        if history:
            current_public["history"] = history[:-1]
        else:
            current_public["history"] = []

        return current_public

    def _tft_responsiveness(self, player_id, public_state):
        opp = self._opponent_id(player_id)
        history = public_state.get("history", [])

        matches = 0
        opportunities = 0

        # Signal immediately before the current stage.
        previous_own_sigma = 0.0

        for h in history:

            active = set(h.get("active", ()))

            opp_sigma = float(
                h["sigma"].get(opp, 0.0)
            )

            # The opponent is responding only when it is the unique
            # active player.
            opponent_is_responder = (
                len(active) == 1
                and opp in active
            )

            if (
                opponent_is_responder
                and previous_own_sigma >= 0.75
            ):
                opportunities += 1

                # Signals are on the grid {0, .25, .5, .75, 1}.
                # Exact matching is the cleanest TFT test.
                if abs(opp_sigma - previous_own_sigma) < 1e-8:
                    matches += 1

            # Current signal becomes the standing signal for the next stage.
            previous_own_sigma = float(
                h["sigma"].get(player_id, previous_own_sigma)
            )

        if opportunities == 0:
            return 0.0

        return float(matches / opportunities)

    def _potential(self, player_id, obs, public_state):
        # -------------------------------------------------------------- #
        # 1. Our current economic position
        # -------------------------------------------------------------- #

        pp = CFG.players[player_id]

        e_hat = float(
            np.clip(obs[E_HAT], 0.0, 1.0)
        )

        rho = float(
            np.clip(obs[RHO], 0.0, 1.0)
        )

        m_hat = float(
            np.clip(obs[M_HAT], 0.0, 1.0)
        )

        K_own = float(obs[K_OWN]) * 4.0
        K_opp = float(obs[K_OPP]) * 4.0

        sigma_own = float(obs[SIG_OWN])
        sigma_opp = float(obs[SIG_OPP])
        mediation = float(obs[MED])

        # Exact exit value.
        X = self._exit_value(
            player_id,
            K_own,
            sigma_opp,
            mediation,
        )

        # Approximate outlasting value.
        B = pp.b0 * (
            pp.b_base
            + pp.b_e * e_hat
            + pp.b_rho * rho
        )

        # Current flow cost.
        g = self._own_flow_cost(
            player_id,
            obs,
        )

        # -------------------------------------------------------------- #
        # 2. Local continuation advantage
        # -------------------------------------------------------------- #
        p_win = self._p_outlasting(player_id, obs, public_state)
        local_margin = p_win * (B - X) - g

        decision_term = np.tanh(
            local_margin / 40.0
        )

        # -------------------------------------------------------------- #
        # 3. Endurance health
        # -------------------------------------------------------------- #

        health_term = (
            0.5 * e_hat
            + 0.3 * rho
            + 0.2 * m_hat
            - 0.5
        )

        # Keep the health contribution relatively small.
        health_term = np.tanh(
            health_term
        )
        # -------------------------------------------------------------- #
        # 4. Commitment
        # -------------------------------------------------------------- #

        # Scale commitment dynamically by the player's actual audience cost
        # parameter (alpha). The U.S. will now properly fear K accumulation.
        commitment_term = (
                                  pp.alpha * K_own
                          ) / 4.0

        # -------------------------------------------------------------- #
        # Escalation Trap
        # -------------------------------------------------------------- #
        total_escalation = (
                sigma_own ** 2
                + (pp.kappa / 3.0) * sigma_opp ** 2
        )

        # If the opponent is responsive, our own escalation is especially
        # dangerous because it is likely to be mirrored.
        tft_like = self._tft_responsiveness(
            player_id,
            public_state,
        )

        responsive_escalation = (
                tft_like * sigma_own ** 2
        )

        mediation = float(obs[MED])

        # -------------------------------------------------------------- #
        # Final potential
        # -------------------------------------------------------------- #
        if player_id == US:
            return float(
                1.00 * decision_term
                + 0.50 * X
                + 0.15 * mediation
                + 0.20 * health_term
                - 0.15 * commitment_term
                - 0.20 * total_escalation
                - 0.30 * responsive_escalation
            )
        elif player_id == IRAN:
            return float(
                1.00 * decision_term
                + 0.50 * B
                + 0.20 * health_term
                - 0.15 * commitment_term
                - 0.20 * total_escalation
                - 0.30 * responsive_escalation
            )

        raise exception("hmm")

        # return float(
        #     1.00 * decision_term
        #     + 0.20 * health_term
        #     - 0.15 * commitment_term
        #     - 0.20 * total_escalation
        #     - 0.30 * responsive_escalation
        # )

    def shaping(self, player_id, obs, action, env_reward, next_obs, next_public, terminated):

        # Recover the public history corresponding to 'obs'.
        current_public = self._current_public_state(
            next_public
        )

        # -------------------------------------------------------------- #
        # Current-state potential
        # -------------------------------------------------------------- #

        phi_now = self._potential(
            player_id,
            obs,
            current_public,
        )

        # -------------------------------------------------------------- #
        # Terminal transition
        # -------------------------------------------------------------- #

        if terminated:
            return float(
                -phi_now
            )

        # -------------------------------------------------------------- #
        # Next-state potential
        # -------------------------------------------------------------- #

        phi_next = self._potential(
            player_id,
            next_obs,
            next_public,
        )

        # -------------------------------------------------------------- #
        # Potential-based shaping
        # -------------------------------------------------------------- #

        return float(
            GAMMA * phi_next - phi_now
        )