"""PetMind AI — educational two-state pet-care simulation.

Three implemented course algorithms:
  1) Bayesian inference (infer hunger from a noisy signal)
  2) Hill climbing (optimize simulated food portion)
  3) One-step MDP/Bellman update (choose FEED vs PLAY)

All probabilities, transitions and rewards are *assumed toy values*, not animal-health guidance.
"""
from __future__ import annotations

import argparse
from typing import Any

PRIOR_HUNGRY = 0.55
SENSOR = {
    "meow": {"hungry": 0.80, "full": 0.25},
    "quiet": {"hungry": 0.20, "full": 0.75},
}
PORTIONS = (1, 2, 3, 4, 5)
GAMMA = 0.70
FUTURE_VALUE = {"hungry": -2.0, "full": 5.0}


def bayesian_hunger(signal: str, prior: float = PRIOR_HUNGRY) -> dict[str, float]:
    """Use Bayes' rule: P(H|o) = P(o|H)P(H) / P(o)."""
    if signal not in SENSOR:
        raise ValueError("Signal must be 'meow' or 'quiet'.")
    if not 0 <= prior <= 1:
        raise ValueError("Prior must be between 0 and 1.")
    likelihood_h = SENSOR[signal]["hungry"]
    likelihood_f = SENSOR[signal]["full"]
    numerator = likelihood_h * prior
    evidence = numerator + likelihood_f * (1 - prior)
    if evidence == 0:
        raise ValueError("Impossible observation under the selected prior.")
    return {"prior": prior, "likelihood_hungry": likelihood_h,
            "likelihood_full": likelihood_f, "evidence": evidence,
            "posterior_hungry": numerator / evidence,
            "posterior_full": 1 - numerator / evidence}


def portion_score(portion: int, p_hungry: float) -> float:
    """Assumed utility of a portion if FEED is selected.

    Reward grows up to 3 portions when hungry; more than 3 is penalized.
    Unnecessary feeding and food cost reduce utility. This is a toy score.
    """
    if portion not in PORTIONS:
        raise ValueError("Portion must be an integer from 1 to 5.")
    if not 0 <= p_hungry <= 1:
        raise ValueError("Hunger probability must be between 0 and 1.")
    benefit = p_hungry * (8 * min(portion, 3) - 4 * max(portion - 3, 0))
    unnecessary = (1 - p_hungry) * 2 * portion
    cost = 0.75 * portion ** 2
    return benefit - unnecessary - cost


def hill_climb_portion(p_hungry: float, start: int = 1) -> dict[str, Any]:
    """Actual discrete hill climbing: check adjacent neighbors, move to best improvement."""
    if start not in PORTIONS:
        raise ValueError("Start must be an integer from 1 to 5.")
    current = start
    trace = []
    while True:
        neighbors = [n for n in (current - 1, current + 1) if n in PORTIONS]
        evaluated = {n: portion_score(n, p_hungry) for n in [current, *neighbors]}
        best = max(evaluated, key=lambda n: (evaluated[n], -n))
        trace.append({"at": current, "neighbors": neighbors,
                      "scores": {n: round(v, 4) for n, v in evaluated.items()},
                      "next": best})
        if best == current:
            break
        current = best
    return {"best_portion": current, "score": portion_score(current, p_hungry),
            "trace": trace, "all_scores": {n: portion_score(n, p_hungry) for n in PORTIONS}}


def transition_full(state: str, action: str, portion: int) -> float:
    """T(Full | state, action). All values are predefined assumptions."""
    if state not in ("hungry", "full") or action not in ("FEED", "PLAY"):
        raise ValueError("Invalid state/action.")
    if action == "FEED":
        return min(0.95, 0.40 + 0.15 * portion) if state == "hungry" else 0.98
    return 0.15 if state == "hungry" else 0.90


def reward(state: str, action: str, portion: int) -> float:
    """One-step action rewards, assumed for the simulation."""
    if action == "FEED":
        return (8 - 0.70 * portion) if state == "hungry" else (-3 - 0.80 * portion)
    return -4.0 if state == "hungry" else 6.0


def bellman_q(state: str, action: str, portion: int) -> dict[str, float]:
    """Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a)*V(s')."""
    p_full = transition_full(state, action, portion)
    expected_future = (1-p_full) * FUTURE_VALUE["hungry"] + p_full * FUTURE_VALUE["full"]
    immediate = reward(state, action, portion)
    return {"reward": immediate, "p_next_full": p_full,
            "future": expected_future, "q": immediate + GAMMA * expected_future}


def mdp_decision(p_hungry: float, portion: int) -> dict[str, Any]:
    """Belief-weighted one-step Bellman action selection (two states, two actions)."""
    details = {}
    for action in ("FEED", "PLAY"):
        hungry = bellman_q("hungry", action, portion)
        full = bellman_q("full", action, portion)
        expected_q = p_hungry * hungry["q"] + (1-p_hungry) * full["q"]
        details[action] = {"hungry": hungry, "full": full, "expected_q": expected_q}
    best = max(details, key=lambda action: details[action]["expected_q"])
    return {"recommended_action": best, "actions": details}


def run_scenario(signal: str, prior: float = PRIOR_HUNGRY) -> dict[str, Any]:
    bayes = bayesian_hunger(signal, prior)
    p = bayes["posterior_hungry"]
    hill = hill_climb_portion(p)
    mdp = mdp_decision(p, hill["best_portion"])
    return {"signal": signal, "bayes": bayes, "hill": hill, "mdp": mdp}


def print_scenario(result: dict[str, Any]) -> None:
    b, h, m = result["bayes"], result["hill"], result["mdp"]
    print("\n" + "="*60)
    print(f"OBSERVATION: {result['signal'].upper()} | prior hungry={b['prior']:.2f}")
    print("="*60)
    print("[1] BAYESIAN INFERENCE")
    print(f"P(hungry | {result['signal']}) = ({b['likelihood_hungry']:.2f} × {b['prior']:.2f}) / "
          f"({b['likelihood_hungry']:.2f} × {b['prior']:.2f} + "
          f"{b['likelihood_full']:.2f} × {1-b['prior']:.2f})")
    print(f"= {b['posterior_hungry']:.4f} ({b['posterior_hungry']:.1%})")
    print("[2] HILL CLIMBING")
    for step in h["trace"]:
        options = ", ".join(f"{n}: {score:.2f}" for n,score in sorted(step['scores'].items()))
        print(f"  At {step['at']}: [{options}] → {step['next']}")
    print(f"Optimal portion if feeding: {h['best_portion']}, score={h['score']:.2f}")
    print("[3] MDP / BELLMAN UPDATE")
    for action, detail in m["actions"].items():
        print(f"  {action}: Q(hungry)={detail['hungry']['q']:.2f}, "
              f"Q(full)={detail['full']['q']:.2f}, "
              f"E[Q]={detail['expected_q']:.2f}")
    print(f"DECISION: {m['recommended_action']}")
    print("Note: predefined probabilities and rewards are educational assumptions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetMind AI - educational AI Fundamentals project")
    parser.add_argument("--signal", choices=["meow", "quiet"])
    parser.add_argument("--prior", type=float, default=PRIOR_HUNGRY)
    args = parser.parse_args()
    signals = [args.signal] if args.signal else ["meow", "quiet"]
    for value in signals:
        print_scenario(run_scenario(value, args.prior))
