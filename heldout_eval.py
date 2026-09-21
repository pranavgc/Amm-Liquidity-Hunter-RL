# =============================================================================
#  HELD-OUT EVALUATION + INDEPENDENT ACCOUNTING CHECKS
#  Companion cell for Finnovator_modified.ipynb
# =============================================================================
#
#  Paste this in as a new cell at the end of the notebook and run it. It depends
#  on nothing from earlier cells except the cloned repo, so it is also safe to
#  run in a fresh runtime straight after cells 0-1.
#
#  Three things it fixes relative to the existing evaluation cells.
#
#  ---------------------------------------------------------------------------
#  1. HELD-OUT SEEDS
#  ---------------------------------------------------------------------------
#  Every seed used so far appears on both sides of the train/test line:
#
#      42     phase-1 training env  AND  scoring seed  AND  rule-agent tuning
#      142    evaluation env during training (seed + 100)
#      242    phase-2 training env (seed + 200)
#      342    phase-3 training env (seed + 300)
#      314    rule-agent tuning seed AND scoring seed
#      2718   rule-agent tuning seed AND scoring seed
#
#  HELDOUT_SEEDS avoids all six. Nothing in it has been trained on or tuned
#  against, so the number it produces is the one worth quoting.
#
#  ---------------------------------------------------------------------------
#  2. FEE ACCOUNTING
#  ---------------------------------------------------------------------------
#  The first evaluation harness in the notebook binds the pre-step fee counters
#  by reference rather than by value:
#
#      f0 = state[LP_COLLECTED_FEES0_KEY]           # aliases the live array
#      state, ... = env.step(action)                # mutates it in place
#      fee_total += state[LP_COLLECTED_FEES0_KEY] - f0      # always exactly 0
#
#  which is why that cell's table reports Fees 0.0 for every agent. Every read
#  below uses .copy().
#
#  ---------------------------------------------------------------------------
#  3. A SANITY CHECK THAT CAN ACTUALLY FAIL
#  ---------------------------------------------------------------------------
#  The existing check defines
#
#      adv_sel := fees - gas - pnl
#
#  and then tests whether  fees - gas - adv_sel - pnl == 0. Substitute the
#  definition and that is identically zero for any inputs at all. It prints a
#  tick regardless of whether the fee, gas or PnL figures are right, so it
#  cannot support the claim it is used to support.
#
#  Below, PnL is measured twice by genuinely independent routes - the summed
#  step rewards, and the change in the simulator's own portfolio value - and
#  the two are compared. That test can fail. Adverse selection is still a
#  residual (the simulator does not expose LVR directly), but it is now a
#  residual between independently measured quantities, which is worth
#  something, and the script says so rather than dressing it up as a proof.
# =============================================================================

import sys, os, json, math, importlib.util
sys.path.insert(0, '.')
import numpy as np

from SAiFE_gym.challenge import ScenarioConfig, create_environment, official_observation
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY, POOL_CURRENT_TICK_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY, PORTFOLIO_VALUE_KEY,
)
from SAiFE_gym.agents.BaselineAgents import HoldToken0Agent, DeployOnceWideAgent


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

INSAMPLE_SEEDS = [42, 314, 2718]              # what the project has been scoring on
HELDOUT_SEEDS  = [7, 1234, 31337, 90210, 555]  # never trained on, never tuned on

# Submission files. Adjust the paths if yours are named differently.
# Entries whose file is missing are skipped with a warning rather than crashing.
SUBMISSIONS = {
    "RL v3 (curriculum)": "my_rl_agent_v3.py",
    "RL v2 (shaped)":     "my_rl_agent_v2.py",
    "Rule-Based CGC":     "my_agent.py",
    "RL v1 (vanilla)":    "my_rl_agent.py",
}
BASELINES = {
    "HoldToken0":     HoldToken0Agent,
    "DeployOnceWide": DeployOnceWideAgent,
}

INITIAL_WEALTH = 1000.0
N_BOOTSTRAP    = 10_000
RNG            = np.random.default_rng(0)

# A gap smaller than this share of the in-sample mean is measurable but not
# worth redrawing conclusions over.
PRACTICAL_GAP_PCT = 10.0


def load_agent_class(path):
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


# ----------------------------------------------------------------------------
# One rollout over one seed
# ----------------------------------------------------------------------------

def rollout(build_agent, seed, cfg):
    """
    Run every trajectory of one seed to termination.

    Returns per-trajectory arrays. PnL is captured two ways:
      pnl_reward  - the sum of the environment's step rewards
      pnl_wealth  - the change in the simulator's own portfolio value
    These come from different places in the state and should agree. Comparing
    them is the check that the tautological one was standing in for.
    """
    env = create_environment(cfg, seed=seed)
    agent = build_agent(env, cfg)

    state, _ = env.reset()
    obs = official_observation(state, env)
    n = cfg.num_trajectories

    pv_start = np.asarray(
        state.get(PORTFOLIO_VALUE_KEY, np.full(n, INITIAL_WEALTH)), dtype=np.float64
    ).copy()

    pnl_reward = np.zeros(n)
    fees       = np.zeros(n)
    n_reb      = np.zeros(n)
    n_inrange  = np.zeros(n)
    widths     = []
    terminated = np.zeros(n, dtype=bool)
    steps      = 0

    while not np.any(terminated):
        action = np.asarray(agent.get_action(obs), dtype=np.float64)

        reb = action[:, 2] <= 0
        n_reb += reb.astype(float)
        if reb.any():
            widths.extend((action[reb, 1] - action[reb, 0]).tolist())

        tick = state[POOL_CURRENT_TICK_KEY]
        lo   = state[LP_TICK_LOWER_KEY]
        hi   = state[LP_TICK_UPPER_KEY]
        n_inrange += ((tick >= lo) & (tick < hi)).astype(float)

        # .copy() is the fix - env.step mutates these arrays in place
        f0_before = state[LP_COLLECTED_FEES0_KEY].copy()
        f1_before = state[LP_COLLECTED_FEES1_KEY].copy()

        state, reward, terminated, _, _ = env.step(action)
        obs = official_observation(state, env)

        pnl_reward += reward
        price = state[ASSET_PRICE_KEY]
        fees += ((state[LP_COLLECTED_FEES0_KEY] - f0_before) * price
                 + (state[LP_COLLECTED_FEES1_KEY] - f1_before))
        steps += 1

    pv_end = np.asarray(
        state.get(PORTFOLIO_VALUE_KEY, np.full(n, np.nan)), dtype=np.float64
    ).copy()

    return dict(
        seed=seed,
        steps=steps,
        pnl_reward=pnl_reward,
        pnl_wealth=pv_end - pv_start,
        fees=fees,
        # The notebook treats the opening deploy as free; both conventions are
        # reported so the choice is visible rather than buried.
        gas_excl_first=np.maximum(n_reb - 1, 0) * cfg.gas_cost,
        gas_incl_first=n_reb * cfg.gas_cost,
        rebalances=n_reb,
        in_range_pct=n_inrange / steps,
        widths=widths,
    )


def evaluate(build_agent, seeds, cfg):
    """Concatenate rollouts across seeds, keeping the per-seed split."""
    per_seed = [rollout(build_agent, s, cfg) for s in seeds]
    keys = ['pnl_reward', 'pnl_wealth', 'fees', 'gas_excl_first',
            'gas_incl_first', 'rebalances', 'in_range_pct']
    out = {k: np.concatenate([r[k] for r in per_seed]) for k in keys}
    out['pnl']      = out['pnl_reward']
    out['gas']      = out['gas_excl_first']
    out['adv_sel']  = out['fees'] - out['gas'] - out['pnl']   # residual, see header
    out['pnl_pct']  = out['pnl'] / INITIAL_WEALTH * 100
    out['per_seed'] = per_seed
    out['widths']   = sum([r['widths'] for r in per_seed], [])
    return out


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def bootstrap_ci(x, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap CI for the mean."""
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def bootstrap_gap_ci(a, b, n_boot=N_BOOTSTRAP, alpha=0.05):
    """
    Percentile bootstrap CI for (mean of b) - (mean of a), resampling each
    group independently. This is the interval to read: if it straddles zero,
    the held-out score is consistent with the in-sample one.
    """
    ia = RNG.integers(0, len(a), size=(n_boot, len(a)))
    ib = RNG.integers(0, len(b), size=(n_boot, len(b)))
    gaps = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    return np.percentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def welch(a, b):
    """Welch's t statistic and approximate two-sided p, no scipy dependency."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    if se == 0:
        return 0.0, 1.0
    t = (a.mean() - b.mean()) / se
    # Normal approximation - sample sizes here are in the hundreds.
    # (math.erf, not np.math - the latter was removed in NumPy 2.0 and Colab
    # currently ships 2.0.2.)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return float(t), float(p)


def summarise(r):
    p = r['pnl']
    lo, hi = bootstrap_ci(p)
    return dict(
        mean=p.mean(), std=p.std(), median=np.median(p),
        min=p.min(), max=p.max(),
        ci_lo=lo, ci_hi=hi,
        profitable=100 * np.mean(p > 0),
        sharpe=p.mean() / (p.std() + 1e-12),
        fees=r['fees'].mean(), gas=r['gas'].mean(),
        adv_sel=r['adv_sel'].mean(),
        rebalances=r['rebalances'].mean(),
        in_range=100 * r['in_range_pct'].mean(),
        n=len(p),
    )


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

cfg = ScenarioConfig()

builders = {}
for name, path in SUBMISSIONS.items():
    if os.path.exists(path):
        A = load_agent_class(path)
        builders[name] = (lambda A: (lambda e, c: A(c.submission_namespace())))(A)
    else:
        print(f"  [skip] {name}: {path} not found")
for name, cls in BASELINES.items():
    builders[name] = (lambda cls: (lambda e, c: cls(e)))(cls)

print(f"\nAgents: {', '.join(builders)}")
print(f"In-sample seeds : {INSAMPLE_SEEDS}  "
      f"({len(INSAMPLE_SEEDS) * cfg.num_trajectories} paths)")
print(f"Held-out seeds  : {HELDOUT_SEEDS}  "
      f"({len(HELDOUT_SEEDS) * cfg.num_trajectories} paths)")
print("\nRunning... (a few minutes)\n")

results = {}
for name, build in builders.items():
    results[name] = {
        'in':  evaluate(build, INSAMPLE_SEEDS, cfg),
        'out': evaluate(build, HELDOUT_SEEDS,  cfg),
    }
    print(f"  done: {name}")


# ----------------------------------------------------------------------------
# Table 1 - in-sample vs held-out
# ----------------------------------------------------------------------------

print("\n" + "=" * 118)
print("HELD-OUT vs IN-SAMPLE".center(118))
print("=" * 118)
print(f"{'Agent':<22} | {'split':<9} | {'mean':>9} | {'95% CI':>19} | "
      f"{'std':>7} | {'min':>9} | {'prof%':>6} | {'sharpe':>6}")
print("-" * 118)

for name, r in results.items():
    for split, label in (('in', 'in-sample'), ('out', 'HELD-OUT')):
        s = summarise(r[split])
        print(f"{name if split == 'in' else '':<22} | {label:<9} | "
              f"{s['mean']:+9.1f} | [{s['ci_lo']:+8.1f}, {s['ci_hi']:+8.1f}] | "
              f"{s['std']:7.1f} | {s['min']:+9.1f} | {s['profitable']:5.1f}% | "
              f"{s['sharpe']:6.2f}")
    print("-" * 118)


# ----------------------------------------------------------------------------
# Table 2 - the generalisation verdict
# ----------------------------------------------------------------------------

print("\n" + "=" * 118)
print("GENERALISATION GAP".center(118))
print("=" * 118)
print("gap = held-out mean minus in-sample mean, with a bootstrapped 95% interval.")
print("If the interval straddles zero, the held-out score is consistent with the")
print("in-sample one and there is no detectable overfitting. A clearly negative gap")
print("means the number currently on the slides is optimistic.\n")
print(f"{'Agent':<22} | {'in-sample':>10} | {'held-out':>10} | {'gap':>9} | "
      f"{'95% CI on gap':>21} | {'gap %':>7} | {'p':>7} | verdict")
print("-" * 118)

for name, r in results.items():
    a, b = r['in']['pnl'], r['out']['pnl']
    gap = b.mean() - a.mean()
    gap_pct = 100 * gap / abs(a.mean()) if a.mean() != 0 else float('nan')
    glo, ghi = bootstrap_gap_ci(a, b)
    _, p = welch(a, b)

    straddles_zero = glo <= 0.0 <= ghi
    if straddles_zero:
        verdict = "holds up"
    elif abs(gap_pct) < PRACTICAL_GAP_PCT:
        verdict = f"measurable but small (<{PRACTICAL_GAP_PCT:.0f}%)"
    elif gap < 0:
        verdict = "OPTIMISTIC in-sample"
    else:
        verdict = "held-out higher - check seed reuse"

    print(f"{name:<22} | {a.mean():+10.1f} | {b.mean():+10.1f} | {gap:+9.1f} | "
          f"[{glo:+9.1f}, {ghi:+9.1f}] | {gap_pct:+6.1f}% | {p:7.4f} | {verdict}")
print("=" * 118)


# ----------------------------------------------------------------------------
# Table 3 - per-seed breakdown
# ----------------------------------------------------------------------------

print("\n" + "=" * 100)
print("PER-SEED MEANS  (a single outlying seed is worth knowing about)".center(100))
print("=" * 100)
for name, r in results.items():
    print(f"\n{name}")
    for split, label in (('in', 'in-sample'), ('out', 'held-out ')):
        parts = [f"{d['seed']}: {d['pnl_reward'].mean():+8.1f}"
                 for d in r[split]['per_seed']]
        print(f"  {label}  " + "   ".join(parts))


# ----------------------------------------------------------------------------
# Accounting checks that can fail
# ----------------------------------------------------------------------------

print("\n" + "=" * 100)
print("INDEPENDENT ACCOUNTING CHECKS".center(100))
print("=" * 100)

print("\nCheck 1 - PnL measured two independent ways")
print("  Route A: sum of the environment's step rewards")
print("  Route B: change in the simulator's own portfolio value (end - start)")
print("  These are read from different places. If they disagree, something in")
print("  the accounting is wrong - this is the test the old one was standing in for.\n")
print(f"  {'Agent':<22} | {'route A':>10} | {'route B':>10} | "
      f"{'mean |A-B|':>11} | {'max |A-B|':>10} | status")
print("  " + "-" * 92)

check1_ok = True
for name, r in results.items():
    a = r['out']['pnl_reward']
    b = r['out']['pnl_wealth']
    if np.all(np.isnan(b)):
        print(f"  {name:<22} | {a.mean():+10.1f} | {'n/a':>10} | "
              f"{'-':>11} | {'-':>10} | portfolio value not exposed")
        continue
    d = np.abs(a - b)
    ok = np.nanmax(d) < 0.01
    check1_ok &= ok
    print(f"  {name:<22} | {a.mean():+10.1f} | {np.nanmean(b):+10.1f} | "
          f"{np.nanmean(d):11.6f} | {np.nanmax(d):10.6f} | "
          f"{'PASS' if ok else 'FAIL - investigate'}")

print("\nCheck 2 - gas reconstructed from the action stream")
print("  Rebalances are counted from the agent's own hold_flag, then priced at")
print("  cfg.gas_cost. Both conventions for the opening deploy are shown.\n")
print(f"  {'Agent':<22} | {'rebalances':>11} | {'gas (excl 1st)':>15} | "
      f"{'gas (incl 1st)':>15} | {'% of stake':>11}")
print("  " + "-" * 92)
for name, r in results.items():
    o = r['out']
    print(f"  {name:<22} | {o['rebalances'].mean():11.1f} | "
          f"{o['gas_excl_first'].mean():15.1f} | {o['gas_incl_first'].mean():15.1f} | "
          f"{100 * o['gas_excl_first'].mean() / INITIAL_WEALTH:10.1f}%")

print("\nCheck 3 - the fee/gas/adverse-selection decomposition")
print("  adverse_selection is a RESIDUAL: fees - gas - PnL. The simulator does")
print("  not expose LVR directly, so this cannot be verified independently here.")
print("  It is reported because check 1 establishes that its three inputs were")
print("  each measured separately - it is NOT a proof of anything on its own.\n")
print(f"  {'Agent':<22} | {'fees':>9} | {'gas':>9} | {'PnL':>9} | "
      f"{'residual':>9} | {'as % fees':>10}")
print("  " + "-" * 92)
for name, r in results.items():
    o = r['out']
    f, g, p = o['fees'].mean(), o['gas'].mean(), o['pnl'].mean()
    res = f - g - p
    pct = 100 * res / f if f != 0 else float('nan')
    print(f"  {name:<22} | {f:9.1f} | {g:9.1f} | {p:+9.1f} | {res:+9.1f} | {pct:9.1f}%")

print("\nCheck 4 - non-degenerate fee capture")
print("  Guards against the .copy() bug silently returning. If any agent that")
print("  rebalances reports exactly zero fees, the harness is broken again.\n")
for name, r in results.items():
    o = r['out']
    if o['rebalances'].mean() > 0 and o['fees'].mean() == 0.0:
        print(f"  {name:<22} FAIL - rebalances but zero fees, check the .copy() reads")
    else:
        print(f"  {name:<22} PASS - fees {o['fees'].mean():.1f}, "
              f"rebalances {o['rebalances'].mean():.1f}")


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------

summary = {
    'insample_seeds': INSAMPLE_SEEDS,
    'heldout_seeds': HELDOUT_SEEDS,
    'paths_per_seed': int(cfg.num_trajectories),
    'gas_cost': float(cfg.gas_cost),
    'initial_wealth': INITIAL_WEALTH,
    'check1_pnl_routes_agree': bool(check1_ok),
    'agents': {},
}
for name, r in results.items():
    summary['agents'][name] = {
        split: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in summarise(r[split]).items()}
        for split in ('in', 'out')
    }

with open("heldout_evaluation.json", "w") as fh:
    json.dump(summary, fh, indent=2)

try:
    import csv
    with open("heldout_evaluation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ['mean', 'std', 'median', 'min', 'max', 'ci_lo', 'ci_hi',
                'profitable', 'sharpe', 'fees', 'gas', 'adv_sel',
                'rebalances', 'in_range', 'n']
        w.writerow(['agent', 'split'] + cols)
        for name, r in results.items():
            for split in ('in', 'out'):
                s = summarise(r[split])
                w.writerow([name, 'in-sample' if split == 'in' else 'held-out']
                           + [f"{s[c]:.4f}" for c in cols])
except Exception as e:      # CSV is a convenience, never fail the cell over it
    print(f"\n(csv write skipped: {e})")

print("\nSaved: heldout_evaluation.json, heldout_evaluation.csv")
print("\nThe held-out mean is the number to put on the slide.")
