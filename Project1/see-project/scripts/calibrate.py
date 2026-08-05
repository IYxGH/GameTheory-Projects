"""Calibration report: episode-length and utility statistics for baseline
matchups under the canonical config. Used to tune parameters so that the
game is non-degenerate (episodes last ~8-25 stages; neither always-continue
nor always-exit dominates; both roles can win)."""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
from see import IRAN, US, PLAYERS, StrategicEnduranceEnv          # noqa: E402
from see.agents import BASELINES                                   # noqa: E402


def play(agent_I, agent_U, seed):
    env = StrategicEnduranceEnv()
    obs, _ = env.reset(seed=seed)
    agent_I.reset(IRAN, seed=seed * 2 + 1)
    agent_U.reset(US, seed=seed * 2 + 2)
    done, steps = False, 0
    while not done:
        acts = {
            IRAN: agent_I.act(obs[IRAN], env.get_legal_mask(IRAN),
                              env.public_state()),
            US: agent_U.act(obs[US], env.get_legal_mask(US),
                            env.public_state()),
        }
        obs, rew, term, _, info = env.step(acts)
        done = term[IRAN]
        steps += 1
    truth = info["_true_state"]
    first_exit = None
    for h in env.public_history:
        if any(h["exit"].values()):
            first_exit = [p for p in PLAYERS if h["exit"][p]]
    return {
        "steps": steps,
        "uI": truth["utility"][IRAN],
        "uU": truth["utility"][US],
        "exiters": tuple(first_exit) if first_exit else ("timeout",),
    }


def main(n=60):
    names = list(BASELINES)
    print(f"{'match':38s} {'len':>10s} {'u_I':>14s} {'u_U':>14s}  outcome mix")
    for nI, nU in itertools.product(names, names):
        res = [play(BASELINES[nI](), BASELINES[nU](), seed=s)
               for s in range(n)]
        steps = np.array([r["steps"] for r in res])
        uI = np.array([r["uI"] for r in res])
        uU = np.array([r["uU"] for r in res])
        from collections import Counter
        mix = Counter(r["exiters"] for r in res).most_common(3)
        mixs = " ".join(f"{'+'.join(k)}:{v}" for k, v in mix)
        print(f"{nI:>16s} vs {nU:16s} "
              f"{steps.mean():5.1f}±{steps.std():4.1f} "
              f"{uI.mean():7.1f}±{uI.std():5.1f} "
              f"{uU.mean():7.1f}±{uU.std():5.1f}  {mixs}")


if __name__ == "__main__":
    main()
