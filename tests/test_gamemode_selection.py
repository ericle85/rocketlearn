"""
Compares three gamemode selection strategies:
  1. equal_episode  — Rolv's equal_likelihood=True path (random choice)
  2. rolv_ema       — Rolv's equal_likelihood=False path (EMA + inv_dist weighting)
  3. least_exp      — Our approach (always pick mode with least cumulative experience)

Run: python tests/test_gamemode_selection.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ---------------------------------------------------------------------------
# Episode simulator
# Avg steps per episode: 1v1~75, 2v2~100, 3v3~120
# PAST_VERSION_PROB: chance one team is old bots (only half agents contribute)
# ---------------------------------------------------------------------------

AVG_STEPS   = {"1v1": 75, "2v2": 100, "3v3": 120}
PAST_VERSION_PROB = 0.2
MODES = ("1v1", "2v2", "3v3")


def simulate_episode(mode: str) -> int:
    """Returns experience (steps * latest_policy_agents) for one episode."""
    b, _ = map(int, mode.split("v"))
    total_agents = b * 2
    steps = max(1, int(np.random.normal(AVG_STEPS[mode], AVG_STEPS[mode] * 0.2)))
    if np.random.random() < PAST_VERSION_PROB:
        latest_agents = b  # one team is old bots
    else:
        latest_agents = total_agents
    return steps * latest_agents


# ---------------------------------------------------------------------------
# Strategy 1: Rolv equal_likelihood=True  (uniform random over modes)
# ---------------------------------------------------------------------------

def select_equal_episode(mode_exp: dict) -> str:
    return np.random.choice(list(mode_exp.keys()))


# ---------------------------------------------------------------------------
# Strategy 2: Rolv equal_likelihood=False
#   Uses EMA of steps-per-episode to normalise target distribution,
#   then weights by how underrepresented each mode currently is.
# ---------------------------------------------------------------------------

def select_rolv_ema(mode_exp: dict, ema: dict) -> str:
    modes = list(mode_exp.keys())
    dist = np.array([mode_exp[m] for m in modes], dtype=float) + 1
    dist /= dist.sum()

    target_dist = np.ones(len(modes))                          # uniform target
    steps_per_ep = np.array([ema.get(m, 1) for m in modes], dtype=float)
    target_dist /= steps_per_ep
    target_dist /= target_dist.sum()

    inv_dist = 1 - dist
    inv_dist /= inv_dist.sum()

    probs = target_dist * inv_dist
    probs /= probs.sum()

    return np.random.choice(modes, p=probs)


# ---------------------------------------------------------------------------
# Strategy 3: Our approach — always pick mode with least experience
# ---------------------------------------------------------------------------

def select_least_exp(mode_exp: dict) -> str:
    min_exp = min(mode_exp.values())
    least = [m for m, v in mode_exp.items() if v == min_exp]
    return np.random.choice(least)


# ---------------------------------------------------------------------------
# Run simulation for a given strategy
# ---------------------------------------------------------------------------

def run_simulation(strategy, n_episodes=2000):
    exp    = {m: 0 for m in MODES}
    counts = {m: 0 for m in MODES}
    ema    = {m: 1 for m in MODES}   # only used by rolv_ema

    for _ in range(n_episodes):
        if strategy == "equal_episode":
            mode = select_equal_episode(exp)
        elif strategy == "rolv_ema":
            mode = select_rolv_ema(exp, ema)
        elif strategy == "least_exp":
            mode = select_least_exp(exp)
        else:
            raise ValueError(strategy)

        ep_exp = simulate_episode(mode)
        exp[mode]    += ep_exp
        counts[mode] += 1

        # update EMA (used by rolv_ema, harmless for others)
        ema[mode] = 0.98 * ema[mode] + 0.02 * ep_exp

    return exp, counts


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_results(label, exp, counts, n_episodes):
    total = sum(exp.values())
    print(f"\n  [{label}]  {n_episodes} episodes")
    print(f"  {'mode':>4}  {'episodes':>9}  {'experience':>11}  {'exp %':>6}  {'deviation':>9}")
    for mode in MODES:
        pct = 100 * exp[mode] / total
        dev = pct - 100/len(MODES)
        print(f"  {mode:>4}  {counts[mode]:>9}  {exp[mode]:>11}  {pct:>5.1f}%  {dev:>+8.1f}%")
    max_dev = max(abs(exp[m]/total - 1/len(MODES)) for m in MODES)
    print(f"  max deviation from equal: {100*max_dev:.1f}%")
    return max_dev


def compare_all():
    print("\n=== Gamemode Selection Strategy Comparison ===")
    print("Simulating 2000 episodes, 1v1 avg=75 steps, 2v2 avg=100, 3v3 avg=120")
    print(f"Past version prob={PAST_VERSION_PROB} (old-bot team halves useful experience)")

    N = 5_000
    results = {}
    for strategy in ("equal_episode", "rolv_ema", "least_exp"):
        np.random.seed(42)
        exp, counts = run_simulation(strategy, N)
        results[strategy] = print_results(strategy, exp, counts, N)

    print("\n  --- Summary (max deviation from 33.3%) ---")
    for s, dev in results.items():
        print(f"  {s:<20} {100*dev:.1f}%")

    # least_exp must beat both Rolv strategies
    assert results["least_exp"] < results["equal_episode"], \
        "least_exp should beat equal_episode"
    assert results["least_exp"] < results["rolv_ema"], \
        "least_exp should beat rolv_ema"
    print("\n  least_exp wins. PASSED")


def test_imbalance_recovery():
    """After a big 3v3 burst, selector should recover to equal distribution."""
    print("\n=== test_imbalance_recovery ===")
    np.random.seed(7)
    exp    = {"1v1": 0, "2v2": 0, "3v3": 100_000}
    counts = {"1v1": 0, "2v2": 0, "3v3": 0}

    for _ in range(500):
        mode = select_least_exp(exp)
        exp[mode] += simulate_episode(mode)
        counts[mode] += 1

    print(f"  3v3 selected {counts['3v3']}/500 times after 100k head start")
    assert counts["3v3"] == 0, f"3v3 should not be picked, got {counts['3v3']}"
    print("  PASSED")


if __name__ == "__main__":
    compare_all()
    test_imbalance_recovery()
    print("\nAll tests done.")
