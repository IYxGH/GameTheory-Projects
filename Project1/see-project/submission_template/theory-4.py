import numpy as np

from see.training.theory_api import TheorySpec
from see.config import canonical_config, IRAN, US


# Observation indices from see/env.py
T_FRAC, RHO, M_HAT, E_HAT, K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = range(9)

# Remaining observations are not needed directly here.
GAMMA = 0.995
CFG = canonical_config()


class MyTheory(TheorySpec):
    """Theory-informed features with conservative Bayesian updating."""

    name = "bayesian-theory"

    # 10 theory features are appended to the original observation.
    extra_feature_dim = 8

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #

    def on_episode_start(self, player_id):
        # No persistent hidden state.
        # Every result is reconstructed from obs + public history, making
        # training and evaluation deterministic for the same information set.
        pass

    # ------------------------------------------------------------------ #
    # Basic helpers
    # ------------------------------------------------------------------ #

    def _opponent_id(self, player_id):
        return US if player_id == IRAN else IRAN

    @staticmethod
    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _signed_tanh(x, scale):
        if scale <= 0.0:
            return float(np.tanh(x))
        return float(np.tanh(float(x) / scale))

    def _exit_value(self, pid, K, sigma_opp, M):
        """Exact public exit value X_i from env.py."""
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
            pp.m_floor
            + (1.0 - pp.m_floor) * min(e_hat, 1.0)
        )

        t = int(round(float(obs[T_FRAC]) * CFG.T_bar))
        ramp = min(1.0, (t + 1) / CFG.ramp_T)

        # env.py charges flow cost using the current standing postures.
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
        """Extract behavior that can legitimately be inferred from history."""
        opp = self._opponent_id(player_id)
        history = public_state.get("history", [])

        if not history:
            return {
                "initiative_high": 0,
                "initiative_count": 0,
                "initiative_mean": 0.0,
                "response_matches": 0,
                "response_count": 0,
                "response_delta_sum": 0.0,
            }

        initiative_high = 0
        initiative_count = 0
        initiative_sum = 0.0

        response_matches = 0
        response_count = 0
        response_delta_sum = 0.0

        # Standing signal of the player immediately before each stage.
        previous_own_sigma = 0.0

        for h in history:
            active = set(h.get("active", ()))
            opp_sigma = float(h.get("sigma", {}).get(opp, 0.0))
            own_sigma = float(h.get("sigma", {}).get(player_id, 0.0))

            # Both players are active: this is the cleanest public evidence
            # about whether the opponent independently chooses escalation.
            if len(active) == 2:
                initiative_count += 1
                initiative_sum += opp_sigma
                if opp_sigma >= 0.75:
                    initiative_high += 1

            # Only the opponent is active: it is responding to us.
            elif len(active) == 1 and opp in active:
                response_count += 1
                response_delta_sum += opp_sigma - previous_own_sigma
                if abs(opp_sigma - previous_own_sigma) < 1e-8:
                    response_matches += 1

            previous_own_sigma = own_sigma

        return {
            "initiative_high": initiative_high,
            "initiative_count": initiative_count,
            "initiative_mean": (
                initiative_sum / initiative_count
                if initiative_count else 0.0
            ),
            "response_matches": response_matches,
            "response_count": response_count,
            "response_delta_sum": response_delta_sum,
        }

    # ------------------------------------------------------------------ #
    # Bayesian behavioral update
    # ------------------------------------------------------------------ #

    def _bayesian_initiative(self, player_id, public_state):
        opp = self._opponent_id(player_id)
        pp = CFG.players[opp]
        stats = self._history_statistics(player_id, public_state)

        prior_rho = pp.rho_a / (pp.rho_a + pp.rho_b)

        # Small prior strength: early observations matter, but not too much.
        prior_strength = 4.0
        alpha0 = 1.0 + prior_strength * prior_rho
        beta0 = 1.0 + prior_strength * (1.0 - prior_rho)

        alpha = alpha0 + stats["initiative_high"]
        beta = beta0 + (
            stats["initiative_count"] - stats["initiative_high"]
        )

        q_high = alpha / (alpha + beta)
        n = stats["initiative_count"]

        # Confidence is 0 with no evidence and approaches 1 smoothly.
        confidence = n / (n + prior_strength + 1.0)

        # This is a Bayesian estimate of behavior
        return (
            float(self._clip01(q_high)),
            float(self._clip01(confidence)),
            float(prior_rho),
        )

    def _estimated_type(self, player_id, public_state):
        opp = self._opponent_id(player_id)
        pp = CFG.players[opp]

        q_high, confidence, prior_rho = self._bayesian_initiative(
            player_id, public_state
        )

        prior_m = pp.m_a / (pp.m_a + pp.m_b)

        # The q_high posterior is on a behavioral probability. We use it as
        # monotone evidence about resolve, with a conservative shrinkage.
        rho_score = prior_rho + 0.85 * confidence * (q_high - prior_rho)
        rho_score = self._clip01(rho_score)

        # m is only weakly identified from initiative behavior. Keep it near
        # its true configured prior mean instead of pretending we know m.
        m_score = prior_m + 0.15 * confidence * (q_high - prior_rho)
        m_score = self._clip01(m_score)

        return rho_score, m_score, q_high, confidence

    # ------------------------------------------------------------------ #
    # Hidden-endurance estimate
    # ------------------------------------------------------------------ #

    def _estimated_opponent_endurance(
        self, player_id, public_state, est_rho, est_m
    ):
        """Reconstruct opponent endurance using only public history."""
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
            sigma_opp = float(h.get("sigma", {}).get(opp, 0.0))
            sigma_own = float(h.get("sigma", {}).get(player_id, 0.0))

            # env.py uses the standing state BEFORE updating K for the next
            # period, so compute current cost first.
            e_hat = max(estimated_e / e0, 0.0)
            m_t = m_bar * (
                pp.m_floor
                + (1.0 - pp.m_floor) * min(e_hat, 1.0)
            )

            ramp = min(1.0, (t + 1) / CFG.ramp_T)

            raw = (
                pp.c0
                + pp.kappa * sigma_own
                + pp.alpha * K_opp
                + pp.eta * sigma_opp * sigma_own
            )

            if m_t > 0.0:
                estimated_e -= ramp * raw / m_t

            # Current action affects next-stage commitment.
            K_opp = CFG.delta_K * K_opp + sigma_opp

        return float(
            self._clip01(max(estimated_e, 0.0) / e0)
        )

    # ------------------------------------------------------------------ #
    # Strategic quantities
    # ------------------------------------------------------------------ #

    def _estimated_opponent_flow_cost(
        self, player_id, obs, est_m, est_e
    ):
        """Estimate opponent's current effective flow cost."""
        opp = self._opponent_id(player_id)
        pp = CFG.players[opp]

        K_opp = float(obs[K_OPP]) * 4.0
        sigma_opp = float(obs[SIG_OPP])
        sigma_own = float(obs[SIG_OWN])

        m_bar = pp.m_lo + (pp.m_hi - pp.m_lo) * est_m
        m_t = m_bar * (
            pp.m_floor
            + (1.0 - pp.m_floor) * min(est_e, 1.0)
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

    def _estimated_time_to_exhaustion(
        self, player_id, obs, public_state, est_rho, est_m, est_e
    ):
        """Return our and opponent's estimated endurance durations."""
        pp_own = CFG.players[player_id]
        rho_own = self._clip01(float(obs[RHO]))
        m_hat_own = self._clip01(float(obs[M_HAT]))
        e0_own = pp_own.E0 * (
            pp_own.e0_base
            + pp_own.e0_rho * rho_own
            + pp_own.e0_m * m_hat_own
        )
        e_own = max(float(obs[E_HAT]), 0.0) * e0_own

        g_own = max(self._own_flow_cost(player_id, obs), 1e-6)
        tau_own = min(e_own / g_own, float(CFG.T_bar))

        opp = self._opponent_id(player_id)
        pp_opp = CFG.players[opp]
        m_bar_opp = pp_opp.m_lo + (pp_opp.m_hi - pp_opp.m_lo) * est_m
        e0_opp = pp_opp.E0 * (
            pp_opp.e0_base
            + pp_opp.e0_rho * est_rho
            + pp_opp.e0_m * est_m
        )
        e_opp = max(est_e, 0.0) * e0_opp

        g_opp = max(
            self._estimated_opponent_flow_cost(
                player_id, obs, est_m, est_e
            ),
            1e-6,
        )
        tau_opp = min(e_opp / g_opp, float(CFG.T_bar))

        return float(tau_own), float(tau_opp)

    def _outlasting_score(self, tau_own, tau_opp):
        """Smooth strategic score based on estimated time advantage."""
        z = np.clip((tau_own - tau_opp) / 4.0, -8.0, 8.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    def _continuation_margin(
        self, player_id, obs, public_state, est_e
    ):
        """Approximate one-step continuation advantage.

        This is deliberately called a *margin*, not the exact dynamic V^C.
        """
        pp = CFG.players[player_id]

        e_hat = self._clip01(float(obs[E_HAT]))
        rho = self._clip01(float(obs[RHO]))

        X = self._exit_value(
            player_id,
            float(obs[K_OWN]) * 4.0,
            float(obs[SIG_OPP]),
            float(obs[MED]),
        )

        B = pp.b0 * (
            pp.b_base
            + pp.b_e * e_hat
            + pp.b_rho * rho
        )

        est_rho, est_m, _, _ = self._estimated_type(
            player_id, public_state
        )
        tau_own, tau_opp = self._estimated_time_to_exhaustion(
            player_id,
            obs,
            public_state,
            est_rho,
            est_m,
            est_e,
        )
        p_proxy = self._outlasting_score(tau_own, tau_opp)
        g = self._own_flow_cost(player_id, obs)

        raw_margin = p_proxy * (B - X) - g
        return self._signed_tanh(raw_margin, 40.0)

    # ------------------------------------------------------------------ #
    # Responsive/TFT behavior
    # ------------------------------------------------------------------ #

    def _response_features(self, player_id, public_state):
        stats = self._history_statistics(player_id, public_state)
        n = stats["response_count"]
        if n == 0:
            return 0.0, 0.0

        match_rate = stats["response_matches"] / n
        mean_delta = stats["response_delta_sum"] / n

        # Squash the signed response delta so it is always bounded.
        delta_score = float(np.tanh(mean_delta / 0.25))
        return float(match_rate), delta_score

    # ------------------------------------------------------------------ #
    # Theory features
    # ------------------------------------------------------------------ #

    def extra_features(self, player_id, obs, public_state):
        """Return 10 theory-derived strategic features."""
        est_rho, est_m, q_high, confidence = self._estimated_type(
            player_id, public_state
        )

        est_e = self._estimated_opponent_endurance(
            player_id,
            public_state,
            est_rho,
            est_m,
        )

        tau_own, tau_opp = self._estimated_time_to_exhaustion(
            player_id,
            obs,
            public_state,
            est_rho,
            est_m,
            est_e,
        )

        # Difference in comparable units: remaining stages of endurance.
        time_advantage = float(np.tanh((tau_own - tau_opp) / 4.0))

        opponent_cost = self._estimated_opponent_flow_cost(
            player_id,
            obs,
            est_m,
            est_e,
        )

        response_match, response_delta = self._response_features(
            player_id, public_state
        )

        margin = self._continuation_margin(
            player_id,
            obs,
            public_state,
            est_e,
        )

        # Exit/outlasting advantage, distinct from the continuation margin:
        # useful because it isolates the terminal-value comparison.
        pp = CFG.players[player_id]
        e_hat = self._clip01(float(obs[E_HAT]))
        rho = self._clip01(float(obs[RHO]))
        B = pp.b0 * (
            pp.b_base
            + pp.b_e * e_hat
            + pp.b_rho * rho
        )
        X = self._exit_value(
            player_id,
            float(obs[K_OWN]) * 4.0,
            float(obs[SIG_OPP]),
            float(obs[MED]),
        )
        exit_gap = self._signed_tanh(B - X, 40.0)

        # Feature vector, all bounded and numerically stable.
        #
        # 0: conservative opponent resolve score
        # 1: conservative opponent m score
        # 2: estimated opponent endurance fraction
        # 3: Bayesian initiative-high posterior mean
        # 4: Bayesian confidence/evidence level
        # 5: own vs opponent time-to-exhaustion advantage
        # 6: estimated opponent current flow cost
        # 7: exact-match responsiveness (TFT-like) -r
        # 8: mean response direction (escalate/de-escalate)
        # 9: continuation/exit terminal-value gap -r
        return np.array(
            [
                est_rho,
                est_m,
                est_e,
                q_high,
                confidence,
                time_advantage,
                float(np.tanh(opponent_cost / 8.0)),
                response_delta,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Potential-based shaping
    # ------------------------------------------------------------------ #

    def _current_public_state(self, next_public):
        """Recover public state corresponding to the current obs."""
        history = next_public.get("history", [])
        current_public = dict(next_public)
        current_public["history"] = history[:-1] if history else []
        return current_public

    def _expected_state_value(self, player_id, obs, public_state, est_e):
        """Approximate absolute expected value of the current state."""
        pp = CFG.players[player_id]

        e_hat = self._clip01(float(obs[E_HAT]))
        rho = self._clip01(float(obs[RHO]))

        X = self._exit_value(
            player_id,
            float(obs[K_OWN]) * 4.0,
            float(obs[SIG_OPP]),
            float(obs[MED]),
        )

        B = pp.b0 * (
                pp.b_base
                + pp.b_e * e_hat
                + pp.b_rho * rho
        )

        est_rho, est_m, _, _ = self._estimated_type(
            player_id, public_state
        )
        tau_own, tau_opp = self._estimated_time_to_exhaustion(
            player_id, obs, public_state, est_rho, est_m, est_e
        )
        p_proxy = self._outlasting_score(tau_own, tau_opp)
        g = self._own_flow_cost(player_id, obs)

        # Calculate absolute expected value: p(win)*B + p(lose)*X - Expected Costs
        expected_value = p_proxy * B + (1.0 - p_proxy) * X - (g * 4.0)

        return self._signed_tanh(expected_value, 100.0)

    def _potential(self, player_id, obs, public_state):
        pp = CFG.players[player_id]

        est_rho, est_m, _, _ = self._estimated_type(player_id, public_state)
        est_e = self._estimated_opponent_endurance(
            player_id, public_state, est_rho, est_m
        )

        tau_own, tau_opp = self._estimated_time_to_exhaustion(
            player_id, obs, public_state, est_rho, est_m, est_e
        )

        time_term = float(np.tanh((tau_own - tau_opp) / 4.0))

        # 1. State Value
        decision_term = self._expected_state_value(
            player_id, obs, public_state, est_e
        )

        # 2. Commitment Penalty (Player-Aware)
        K_own = float(obs[K_OWN]) * 4.0
        commitment_term = (pp.alpha * K_own) / 4.0

        # 3. Escalation Trap (Includes Joint Risk)
        sigma_own = float(obs[SIG_OWN])
        sigma_opp = float(obs[SIG_OPP])

        total_escalation = (
                sigma_own ** 2
                + (pp.kappa / 3.0) * sigma_opp ** 2
                + (pp.eta / 2.5) * (sigma_own * sigma_opp)
        )

        # 4. Mediation Attractor
        mediation = float(obs[MED])

        return float(
            0.80 * decision_term
            + 0.15 * time_term
            - 0.15 * commitment_term
            - 0.20 * total_escalation
            + 0.15 * mediation
        )

    def shaping(
        self,
        player_id,
        obs,
        action,
        env_reward,
        next_obs,
        next_public,
        terminated,
    ):
        """Potential-based shaping: F = gamma Phi(next) - Phi(now)."""
        current_public = self._current_public_state(next_public)

        phi_now = self._potential(
            player_id,
            obs,
            current_public,
        )

        # Terminal states use Phi = 0.
        if terminated:
            return float(-phi_now)

        phi_next = self._potential(
            player_id,
            next_obs,
            next_public,
        )

        return float(GAMMA * phi_next - phi_now)
        # return 0.0