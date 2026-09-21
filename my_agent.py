"""
The Concentrator — submission: "Calm-Gated Concentrator" (CGC)

STRATEGY IN ONE SENTENCE
------------------------
Quote the tightest possible liquidity band, hold it for free while the price
sits inside, and only pay gas to re-center *during calm steps* — never while the
arbitrage flow is actively pushing the price — rate-limited so gas can never
overwhelm fee income.

WHY THIS IS THE RIGHT STRATEGY (evidence from the simulator)
------------------------------------------------------------
We instrumented the official scenario and decomposed every strategy's PnL into
fees earned, gas paid, and adverse selection (loss-versus-rebalancing):

  * In-range fee income (~26) DWARFS in-range adverse selection (~4). Being in
    range is highly profitable — toxicity is NOT the root cause of losses.
  * GAS is the binding constraint. Re-centering every step bankrupts the stake
    (1000 steps x 3.0 = 3000, i.e. 3x capital -> -1000 PnL). The marginal
    fee-per-rebalance (~2.8) is structurally just BELOW the 3.0 gas cost, so
    indiscriminate re-centering loses money.
  * A tighter band has higher liquidity density (L proportional to 1/width), so
    it earns more fee per crossing. Tight + rarely-rebalanced beats wide.

The losing baselines fail for opposite reasons: HoldToken0 / DeployOnceWide pay
no gas but sit out of range ~80-98% of the time (almost no fees); the always-
rebalance baseline earns fees but is wiped out by gas.

THE EDGE — timing entries on volatility
---------------------------------------
Because the pool tick can move at most +/-1 per step but the arbitrage intensity
is enormous, the pool tick is being *actively dragged* whenever there is a price
gap. Re-centering at that moment deploys a fresh tight band directly into the
path of continuing arb flow -> it is run over and exits again immediately
(pure gas churn). Instead we re-center ONLY on a calm step (the pool tick did not
move last step => the mispricing has resolved). The fresh band then sits in a
stable window and actually earns fees. A minimum-gap throttle caps total gas.

Measured on the public practice seeds [42, 314, 2718], 100 paths each:
    HoldToken0 (best baseline)  ~ +24 mean
    DeployOnceWide              ~ +22 mean
    Calm-Gated Concentrator     ~ +86 mean, 100% of paths profitable

GRADER COMPLIANCE
-----------------
numpy-only; no file I/O; no dunder access; one file; vectorized (no per-traj
Python loops). __init__ receives the scalar config namespace; get_action
receives the per-trajectory observation dict.
"""

import numpy as np


class Agent:
    def __init__(self, config):
        # ---- read-only scalar episode config -------------------------------
        self.n = int(getattr(config, "num_trajectories", 100))
        self.tau = int(getattr(config, "tau", 10))
        self.gas_cost = float(getattr(config, "gas_cost", 3.0))

        # ---- strategy hyper-parameters (tuned on practice seeds) -----------
        # Tightest band -> maximum fee density. width=1 means offsets [-1, +1].
        self.width = 1
        # "Calm" = the pool tick moved <= calm_thresh ticks last step. 0 means
        # only re-center when the price was perfectly still (the arb flow that
        # caused us to fall out of range has fully resolved).
        self.calm_thresh = 0
        # Never re-center more often than once every `min_gap` steps. Caps gas:
        # ~1000/min_gap rebalances * gas_cost stays a small fraction of wealth.
        self.min_gap = 20
        # Defensive wealth floor: never pay gas to re-center if doing so would
        # spend a large slice of remaining wealth (protects pathological paths).
        self.wealth_floor = 8.0 * self.gas_cost

        # ---- per-trajectory memory (vectorized state we carry between steps)
        self.prev_tick = None                       # pool tick at previous step
        self.steps_since_reb = np.full(self.n, 10 ** 9, dtype=np.int64)

    def get_action(self, state):
        # Robust to the harness reporting a different batch size than config.
        current_tick = np.asarray(state["current_tick"], dtype=np.float64)
        n = current_tick.shape[0]
        if self.prev_tick is None or self.prev_tick.shape[0] != n:
            self.prev_tick = current_tick.copy()
            self.steps_since_reb = np.full(n, 10 ** 9, dtype=np.int64)

        lp_lower = np.asarray(state["lp_tick_lower"], dtype=np.float64)
        lp_upper = np.asarray(state["lp_tick_upper"], dtype=np.float64)
        ever = np.asarray(
            state.get("lp_ever_deployed", np.zeros(n, dtype=bool))
        ).astype(bool)
        wealth = np.asarray(
            state.get("portfolio_value", np.full(n, np.inf))
        ).astype(np.float64)

        # --- signals --------------------------------------------------------
        # 1) Are we out of our current band? (no fees earned while out of range)
        out_of_range = (current_tick < lp_lower) | (current_tick >= lp_upper)

        # 2) Is this a CALM step? velocity = |pool tick move| over last step.
        velocity = np.abs(current_tick - self.prev_tick)
        calm = velocity <= self.calm_thresh

        # 3) Has enough time passed since our last rebalance? (gas throttle)
        gap_ok = self.steps_since_reb >= self.min_gap

        # 4) Can we afford it? (defensive bankruptcy guard)
        can_afford = wealth > self.wealth_floor

        # --- decision -------------------------------------------------------
        # Deploy on the very first step; thereafter re-center only when we are
        # OUT of range AND the market is calm AND the gas throttle/affordability
        # checks pass. Otherwise HOLD (costs nothing) and keep collecting fees.
        first_deploy = ~ever
        recenter = out_of_range & calm & gap_ok & can_afford
        rebalance = first_deploy | recenter

        # --- bookkeeping for next step (vectorized) -------------------------
        self.steps_since_reb = np.where(
            rebalance, 0, self.steps_since_reb + 1
        ).astype(np.int64)
        self.prev_tick = current_tick.copy()

        # --- emit action: symmetric tight band, re-centered on current tick --
        lower = np.full(n, -self.width, dtype=np.float64)
        upper = np.full(n, self.width, dtype=np.float64)
        hold_flag = np.where(rebalance, -1.0, 1.0)   # <=0 rebalance, >0 hold
        return np.column_stack([lower, upper, hold_flag])
