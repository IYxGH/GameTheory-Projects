"""Environment invariants. Run with: python -m pytest tests/ -q"""
import numpy as np
import pytest

from see import (ACTION_EXIT, ACTION_HOLD, IRAN, US, PLAYERS,
                 StrategicEnduranceEnv, canonical_config)
from see.config import HIGH_SIGNAL
from see.agents import RandomAgent


def rollout(env, policy, seed=0, max_steps=200):
    """Play an episode; return per-player summed rewards and final info."""
    obs, info = env.reset(seed=seed)
    totals = {p: 0.0 for p in PLAYERS}
    for _ in range(max_steps):
        actions = {p: policy(p, obs[p], env.get_legal_mask(p)) for p in PLAYERS}
        obs, rew, term, trunc, info = env.step(actions)
        for p in PLAYERS:
            totals[p] += rew[p]
        if term[IRAN]:
            break
    return totals, info


def random_policy_factory(seed=0):
    rng = np.random.default_rng(seed)

    def pol(pid, obs, mask):
        return int(rng.choice(np.flatnonzero(mask)))
    return pol


def test_reset_shapes_and_privacy():
    env = StrategicEnduranceEnv(seed=1)
    obs, info = env.reset(seed=1)
    for p in PLAYERS:
        assert obs[p].shape == (16,)
    # own private fields present, opponent's absent by construction:
    # obs contains own rho/m/e, and public K/sigma only for opponent
    oI, oU = obs[IRAN], obs[US]
    truth = info["_true_state"]
    assert abs(oI[1] - truth["rho"][IRAN]) < 1e-6
    assert abs(oU[1] - truth["rho"][US]) < 1e-6
    # no field in Iran's obs equals U.S. rho or e (privacy smoke check)
    assert not np.any(np.isclose(oI, truth["rho"][US]))


def test_seed_reproducibility():
    e1 = StrategicEnduranceEnv()
    e2 = StrategicEnduranceEnv()
    t1, _ = rollout(e1, random_policy_factory(7), seed=42)
    t2, _ = rollout(e2, random_policy_factory(7), seed=42)
    assert t1 == t2


def test_return_equals_paper_utility():
    """Undiscounted return must equal u_i = terminal - G - sum(phi)."""
    env = StrategicEnduranceEnv()
    totals, info = rollout(env, random_policy_factory(3), seed=5)
    truth = info["_true_state"]
    for p in PLAYERS:
        assert totals[p] == pytest.approx(truth["utility"][p], abs=1e-6)
        # utility = terminal value - G - phi_spend  =>  terminal recoverable
        terminal = totals[p] + truth["G"][p] + env.phi_spend[p]
        assert np.isfinite(terminal)


def test_forced_exit_when_endurance_depleted():
    env = StrategicEnduranceEnv()
    env.reset(seed=0)
    env.e[IRAN] = -1.0
    mask = env.get_legal_mask(IRAN)
    assert mask[ACTION_EXIT] and mask.sum() == 1


def test_hold_forced_for_non_mover():
    """A unilateral high signal must trigger a response turn."""
    env = StrategicEnduranceEnv()
    env.reset(seed=0)
    # Iran escalates hard, U.S. stays low -> next stage only U.S. moves
    env.step({IRAN: 4, US: 0})
    if not env.exited[IRAN]:
        assert env.get_active_movers() == [US]
        mI = env.get_legal_mask(IRAN)
        assert mI[ACTION_HOLD] and mI.sum() == 1
        # held signal persists as public posture
        assert env.last_sigma[IRAN] == 1.0


def test_exit_ends_episode_and_pays_B_and_X():
    env = StrategicEnduranceEnv()
    env.reset(seed=11)
    obs, rew, term, trunc, info = env.step({IRAN: ACTION_EXIT, US: 2})
    assert term[IRAN] and term[US]
    # exiter reward includes X (bounded), survivor includes B > 0 net of costs
    assert rew[US] > rew[IRAN]


def test_timeout_pays_exit_values():
    env = StrategicEnduranceEnv()
    env.reset(seed=2)
    env.cfg.e_noise = 0.0
    done = False
    steps = 0
    while not done and steps < 100:
        acts = {}
        for p in PLAYERS:
            m = env.get_legal_mask(p)
            acts[p] = 0 if m[0] else int(np.argmax(m))  # never volunteer exit
        _, _, term, _, _ = env.step(acts)
        done = term[IRAN]
        steps += 1
    assert done  # either timeout at T_bar or forced exit by depletion
    assert steps <= env.cfg.T_bar


def test_single_crossing_of_signaling_cost():
    """phi must be increasing in sigma and its sigma-slope decreasing in e."""
    cfg = canonical_config()
    pp = cfg.players[IRAN]

    def phi(sigma, rho, e_hat, m_hat):
        return (pp.phi0 * sigma ** 2
                / (pp.phi_w + (1 - pp.phi_w) * rho)
                / (pp.phi_w + (1 - pp.phi_w) * e_hat)
                / (pp.phi_v + (1 - pp.phi_v) * m_hat))

    for rho in (0.2, 0.8):
        for m in (0.1, 0.9):
            # increasing in sigma
            assert phi(1.0, rho, 0.5, m) > phi(0.5, rho, 0.5, m) > 0
            # decreasing in e, rho, m
            assert phi(1.0, rho, 0.9, m) < phi(1.0, rho, 0.3, m)
    # single crossing: marginal cost of signal lower for high-e types
    slope_low_e = phi(1.0, 0.5, 0.3, 0.5) - phi(0.75, 0.5, 0.3, 0.5)
    slope_high_e = phi(1.0, 0.5, 0.9, 0.5) - phi(0.75, 0.5, 0.9, 0.5)
    assert slope_high_e < slope_low_e


def test_masks_reject_illegal_actions_gracefully():
    env = StrategicEnduranceEnv()
    env.reset(seed=3)
    env.step({IRAN: 4, US: 0})           # trigger response turn (US moves)
    if not env.exited[IRAN] and env.get_active_movers() == [US]:
        # Iran illegally tries to exit during U.S. response turn -> coerced HOLD
        obs, rew, term, trunc, info = env.step({IRAN: ACTION_EXIT, US: 1})
        assert not env.exited[IRAN]


def test_random_agents_full_episodes():
    """Smoke: 50 random-vs-random episodes terminate with finite utilities."""
    env = StrategicEnduranceEnv()
    aI, aU = RandomAgent(), RandomAgent()
    for ep in range(50):
        obs, _ = env.reset(seed=1000 + ep)
        aI.reset(IRAN, seed=ep)
        aU.reset(US, seed=10_000 + ep)
        done = False
        while not done:
            acts = {
                IRAN: aI.act(obs[IRAN], env.get_legal_mask(IRAN)),
                US: aU.act(obs[US], env.get_legal_mask(US)),
            }
            obs, rew, term, _, info = env.step(acts)
            done = term[IRAN]
        for p in PLAYERS:
            assert np.isfinite(info["_true_state"]["utility"][p])
