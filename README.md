# AMM Liquidity Optimisation

Reinforcement learning for concentrated-liquidity placement in a simulated Uniswap V3 pool.

Built in 48 hours for the **TurinTech AI "SaiFE Sandbox" challenge** at the UKFinnovator Innovation
Competition, Imperial College London, 13–14 June 2026. **Second place in the challenge group.**

A curriculum-trained PPO agent returns **+48.02%** on a 1,000-token stake in simulation, against
**+2.44%** for the strongest baseline — 19.7× — and is profitable on all 300 evaluation paths.

---

## The problem

You are the sole liquidity provider in a Uniswap V3-style pool over 1,000 steps. Each step you either
hold your position or pay gas to re-centre it. Three forces decide whether you make money:

| Force | Detail |
|---|---|
| **Fee income** | Earned only while the pool tick sits inside your band. Tighter band → higher liquidity density (L ∝ 1/width) → more fee per crossing. |
| **Gas** | Flat 3.0 tokens per rebalance. Re-centring every step costs 1,000 × 3.0 = 3,000, three times the entire stake. |
| **Adverse selection** | Value lost to arbitrageurs pulling the pool price back toward the external midprice. |

The tension is obvious once you write it down: a tight band earns well but falls out of range quickly,
and chasing it with rebalances costs more than it earns. The whole problem is deciding *when* to move.

## What we found first

We instrumented the simulator before writing any policy, and the decomposition contradicted the
intuitive read:

- **In-range fee income (~26) dwarfs in-range adverse selection (~4).** Toxic flow is not the binding
  constraint. Coverage is.
- **The full-range rebalance-every-step baseline scores exactly −1000.00 on every path** — a total
  wipeout. Gas is what kills you, not arbitrageurs.
- **The best baseline is doing nothing.** HoldToken0 scores +24.42 and loses money on 11.7% of paths.
  Beating its mean is not enough; the tail matters.

That reframing — from "avoid toxic flow" to "maximise range coverage without bleeding gas" — drove
everything below.

---

## Results

Official grader, practice mode, 3 seeds × 100 trajectories = 300 paths of 1,000 steps each.

| Agent | Mean PnL | Return | σ | Median | Min | Max | Profitable | Sharpe |
|---|---|---|---|---|---|---|---|---|
| **RL v3 (curriculum PPO)** | **+480.23** | **+48.02%** | 124.22 | +475.04 | +128.62 | +851.89 | **100.0%** | 3.87 |
| Rule-based CGC | +86.50 | +8.65% | 31.4 | — | +10.30 | +196.00 | 100.0% | 2.75 |
| RL v2 (shaped reward) | +55.80 | +5.58% | 28.2 | — | — | — | 98.3% | — |
| RL v1 (vanilla PPO) | +24.42 | +2.44% | 22.47 | +21.18 | −26.30 | +102.24 | 88.3% | 1.09 |
| HoldToken0 *(baseline)* | +24.42 | +2.44% | 22.47 | +21.18 | −26.30 | +102.24 | 88.3% | 1.09 |
| DeployOnceWide *(baseline)* | +21.47 | +2.15% | 18.95 | +20.45 | −17.16 | +80.40 | 87.7% | 1.13 |
| FullRangeRebalance *(baseline)* | −1000.00 | −100.00% | 0.00 | −1000.00 | −1000.00 | −1000.00 | 0.0% | — |

All figures on a 1,000-token initial stake. Blank cells were not recorded in the run that produced
that row.

### What the winning agent actually does

This is the part that surprised us, and it is worth stating plainly because it contradicts our own
opening thesis.

| Agent | Rebalances | Gas paid | Fees earned | Fee per rebalance | In range |
|---|---|---|---|---|---|
| RL v3 | 291 | 870.8 | 1543.4 | 5.30 | 32.6% |
| Rule-based CGC | 31 | 89.3 | 195.7 | 6.31 | 12.0% |
| HoldToken0 | 1 | 0.0 | 28.7 | — | 1.2% |

v3 spends **87% of the entire initial stake on gas** and wins anyway. Our early measurement said the
marginal fee per rebalance (≈2.8) sat just below the 3.0 gas cost, so we built the rule-based agent
around restraint. That measurement was taken over *indiscriminate* rebalancing, including into active
arbitrage flow where a fresh band is immediately run over. Place the band well and the economics
invert: both our agents clear 5–6 in fees per rebalance against 2.99 in gas.

The conservative reading was not wrong about gas. It was wrong about which rebalances cost you.

One more result worth recording: **adverse selection scales with fee income, not with trading
frequency** — 12.5% of fees for v3, 10.2% for the rule-based agent, ~15% for the do-nothing baselines.
Rebalancing nine times more often did not increase per-unit exposure to arbitrageurs.

---

## How we got there

Four agents, each built because the previous one failed in a way we could name.

**1. Calm-Gated Concentrator** (`agents/rule_based_cgc.py`) — no learned parameters. One boolean:

```python
rebalance = first_deploy | (out_of_range & calm & gap_ok & can_afford)
#   calm       : |Δ pool tick| == 0 last step (the mispricing has resolved)
#   gap_ok     : >= 20 steps since the last rebalance (gas throttle)
#   can_afford : wealth > 8 * gas_cost
```

The pool tick moves at most ±1 per step, but arbitrage intensity is high, so whenever a price gap
exists the tick is being actively dragged. Re-centring at that moment drops a fresh tight band into
continuing arb flow and it exits range immediately — pure gas churn. Waiting for a still tick means
the band lands in a stable window. **+86.5, profitable on 100% of paths, 31 rebalances.**

**2. RL v1 — vanilla PPO** (`agents/rl_agent_v1.py`) — 5 features, 4,739 parameters. Collapsed to a
degenerate hold policy. Its evaluation row is *identical to HoldToken0 in all twelve reported
metrics*: mean, σ, min, max, Sharpe, fees, gas, rebalance count, in-range fraction. The agent had
learned to never pay gas. With undiscounted returns and a sparse reward, inaction is the safe gradient
direction. **+24.4 — a tie with doing nothing.**

**3. RL v2 — shaped reward** (`agents/rl_agent_v2.py`) — 6 features, 4,803 parameters. Three changes,
one per failure mode: a dense in-range bonus so fee capture is visible before the agent assembles a
full strategy, an anti-HODL penalty growing with time out of range, and `ent_coef` raised to 0.05 to
keep the policy exploring. Added tick velocity to the observation — the calm signal the rule-based
agent used explicitly. **+55.8, 98.3% profitable, 14.1% in range.**

**4. RL v3 — curriculum PPO** (`agents/rl_agent_v3.py`) — 8 features, 4,931 parameters. Trained in
three phases with gas ramped `0 → 1.5 → 3.0`. At zero gas, rebalancing is free, so the agent learns
fee capture and band placement without ever being punished for acting. Gas is then introduced
gradually and it learns to ration a behaviour it already knows is valuable, rather than learning to
fear one it never tried. **+480.2 — 8.6× v2, 19.7× the best baseline.**

The training logs show the mechanism directly: v2's shaped episode reward opened around −3,070 and
spent its early budget climbing out of a hole; v3's phase 1 opened at +607 and rose from there.

---

## Repository layout

```
.
├── agents/
│   ├── rule_based_cgc.py        # Calm-Gated Concentrator, no learned parameters
│   ├── rl_agent_v1.py           # vanilla PPO, 5 obs   (collapsed to HODL)
│   ├── rl_agent_v2.py           # shaped reward, 6 obs
│   └── rl_agent_v3.py           # curriculum PPO, 8 obs  <-- submitted
├── notebooks/
│   └── Finnovator_modified.ipynb  # full training pipeline, outputs preserved
├── eval/
│   └── heldout_eval.py          # held-out evaluation + independent accounting checks
├── figures/
│   ├── pnl_distribution.png
│   └── baseline_comparison.png
└── slides/
    ├── DeFi-Strategy-Pitch.pptx
    └── AMM-Liquidity-Optimisation.pptx
```

Each agent file is standalone and dependency-free beyond NumPy — that was a submission constraint
(one file, no file I/O, no dunder access, 64 KB limit). The trained policies carry their weights
inlined as comma-separated `%.4e` literals parsed by `np.fromstring`. v3 ships at 58.7 KiB.

---

## Reproducing

The notebook runs top to bottom on a Colab T4 and regenerates both trained agents.

```bash
git clone https://github.com/giorgoschionas/concentrator-challenge.git
cd concentrator-challenge
pip install stable-baselines3 gymnasium numpy
```

Then open `notebooks/Finnovator_modified.ipynb`. Cells 0–43 build v2; cells 44–53 build v3.

Score a submission against the official grader:

```bash
python experiments/hackathon_simulation.py --agent agents/rl_agent_v3.py --include-full-range
```

Versions we ran on: Stable-Baselines3 2.8.0, Gymnasium 1.2.3, PyTorch 2.11.0+cu128, NumPy 2.0.2.

### Training configuration

| | v2 | v3 |
|---|---|---|
| Observations | 6 | 8 |
| Network | 64 × 64 tanh, pi and vf | 64 × 64 tanh, pi and vf |
| Learning rate | 3e−4 | 3e−4 |
| `gamma` | 1.0 | 1.0 |
| `gae_lambda` / `clip_range` | 0.95 / 0.2 | 0.95 / 0.2 |
| `ent_coef` | 0.05 | 0.02 |
| Rollout / batch | 100,000 / 6,250 | 100,000 / 6,250 |
| Gas during training | 3.0 (constant) | 0.0 → 1.5 → 3.0 |
| Timesteps | 3M | ~5.4M (see Known issues) |

The evaluation environment is always built with reward shaping disabled and gas at the full 3.0, so
model selection was never judged on the shaped signal.

### Observation space (v3)

`mispricing`, `boundary_proximity`, `position_width`, `time`, `gas_cost`, `tick_velocity`,
`rolling_vol` (σ of velocity over a 20-step circular buffer), `time_since_rebalance` (capped at 100
steps, normalised).

Features 6–8 were added deliberately: velocity is the calm signal the rule-based agent hand-coded,
and the last two are the two constants that agent encoded as fixed hyper-parameters (`min_gap`, the
calm threshold), handed to the policy so it could set them itself.

---

## Limitations

We would rather state these than have someone find them.

- **Synthetic throughout.** GBM plus Poisson arrivals. No real price shocks, no news events, no gas
  price volatility, no fat tails. A strategy that spends 87% of the stake on gas is unusually
  sensitive to the gas model, and the gas model here is a constant.
- **No held-out evaluation in the original submission.** Seeds `[42, 314, 2718]` were used for scoring;
  42 is also the phase-1 training seed, and the rule-based agent's four constants were tuned on those
  same three seeds. `eval/heldout_eval.py` fixes this — run it on fresh seeds before quoting the 48%
  anywhere that matters.
- **Probable overfitting to the simulator.** 100% of paths profitable at Sharpe 3.87 is extraordinary
  for a market-making strategy. The most economical explanation is that the agent has learned the
  generative process, not a market edge.
- **Single pool, single position.** The simulator allows one band, so multi-range strategies were
  never testable.
- **No execution risk.** Rebalances are assumed to succeed. At 291 transactions per episode, real
  failure rates, slippage and front-running would compound — and calm-gating deliberately acts at
  predictable moments, which may be exactly when front-running is easiest.

## Known issues

- **Actual v3 training was ~5.4M timesteps, not the 3M intended.** The three phases call
  `learn(800_000 / 1_600_000 / 3_000_000, reset_num_timesteps=False)`, and SB3 treats
  `total_timesteps` as a cumulative target under that flag, so each phase runs that many *additional*
  steps. Confirmed against the shipped weights: the gas normalisation statistics
  (`obs_mean[4] = 2.1111`, `obs_var[4] = 1.2099`) match the 800k/1.6M/3.0M weighting to six decimal
  places, where the intended split would give 1.8 / 1.56.
- **The original accounting check was a tautology.** `adv_sel` is defined as `fees − gas − pnl`, then
  checked with `fees − gas − adv_sel − pnl == 0`, which is identically zero for any input. It cannot
  fail. `eval/heldout_eval.py` replaces it with a comparison of PnL measured two independent ways
  (summed step rewards vs. change in the simulator's own portfolio value), which can.
- **Fee accounting bug in the first evaluation harness** (notebook cell 35): pre-step fee counters are
  bound by reference rather than `.copy()`, and `env.step` mutates them in place, so every fee delta
  reads as zero. That cell's table reports Fees 0.0 for every agent. Cells 48 and 51 are correct and
  are the source of the numbers above.
- **The shipped v3 is the final model, not the best checkpoint.** `best_models_v3/best_model.zip` was
  not found at extraction time, so `EvalCallback`'s selection was never used.
- **`CurriculumCallback` is dead code.** It only prints and is never passed to `learn()`. The
  curriculum works by rebuilding the environment between the three `learn()` calls and deep-copying
  `obs_rms` forward.
- **v1 and v2 cannot generalise across gas regimes.** Gas was constant at 3.0 during their training,
  so the recorded variance is ~5e−10; any other gas value saturates the ±10 normalisation clip and the
  feature degenerates to a binary flag. v3 does not have this problem.

## Next steps

1. Run `eval/heldout_eval.py` on unseen seeds and requote the headline number from that.
2. Stress-test across gas regimes and gas volatility.
3. Model execution failure and front-running.
4. Backtest on real Uniswap V3 tick data.
5. Hybrid design: RL for the rebalance decision, the rule-based agent as a safety filter on the action.

---

## Team

Three-person team, 48 hours. All three of us worked across every agent and the evaluation.

- *Aditya Narayan Singh* — [https://github.com/…](https://github.com/LiveAdityaSingh)
- *Chandrabhushan Prasad* — [https://github.com/…](https://github.com/CbPrasad07)
- *Pranav Ganesh Chandratre* — [https://github.com/…
](https://github.com/pranavgc)
## Acknowledgements

Challenge and simulator by [TurinTech AI](https://www.turintech.ai/) —
[`giorgoschionas/concentrator-challenge`](https://github.com/giorgoschionas/concentrator-challenge).
Hosted by UKFin+ and Imperial College London.

## Licence

*MIT License*
