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

    name = "my-theory"

    # Number of extra belief features you append each stage. Set this to
    # the length of the vector extra_features() returns. Leave 0 for the
    # null theory below.
    extra_feature_dim = 0

    def on_episode_start(self, player_id):
        """Called once at the start of every episode. Reset any per-episode
        state here (e.g. a belief filter you carry across stages)."""
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
        # Example of the MECHANICS (delete and write your own):
        #     feats = np.zeros(self.extra_feature_dim, dtype=np.float32)
        #     feats[0] = ...      # a statistic YOU derive in Section 4
        #     return feats
        return np.zeros(self.extra_feature_dim, dtype=np.float32)

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
        GAMMA = 0.995

        if player_id == "US":
            if obs[8] == 1:
                if action == 5:
                    return 1000
                else:
                    return -1000
            else:
                if action == 0:
                    return 100
                else:
                    return -100



        return 0.0
