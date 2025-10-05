"""
Enhanced trade simulation with:
- independent win probability per trade (Bernoulli)
- variable RRR per trade
- slippage & commissions
- risk cap (max dollar risk per trade)
- dynamic risk reduction after large drawdowns
- per-trade vs deferred tax option
- Monte Carlo multiple-run capability
- summary statistics (expectancy, profit factor, sharpe-like metric)
- visualizations (equity curve sample + histogram of final balances)
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Dict
import statistics

# ---------------------------
# Configurable parameters
# ---------------------------
STARTING_BALANCE = 25_000
NUM_TRADES = 250
SEED = 23

# Trade characteristics
WIN_PROB = 0.28                # mean expected win probability (25% - 33% typical)
RRR_LOW, RRR_HIGH = 3.0, 30.0  # reward-to-risk random range per trade

# Costs & realism
COMMISSION_PER_TRADE = 2.5     # flat commission per trade ($)
SLIPPAGE_PCT = 0.001           # slippage as percentage of reward/loss (0.1% default)

# Position sizing
RISK_PCT = 0.01                # fraction of equity risked per trade
MAX_RISK_DOLLARS = 2_000       # cap risk per trade in USD (set to None to disable)
REDUCE_RISK_AFTER_DD_PCT = 20  # if drawdown > this percent, reduce risk
REDUCED_RISK_PCT = 0.005       # reduced risk percent after big drawdown

# Taxation
CALC_TAX = True
TAX_PCT = 30                   # percent on realized profits if per-trade tax
DEFER_TAX_TO_END = False       # if True, apply tax on net profit at simulation end

# Reporting / Monte Carlo
REPORT_INTERVAL = 50
REPORT_TRADES = False
NUM_MONTE_CARLO_RUNS = 500     # set >1 to run multiple sims for distribution

# Output / plotting
PLOT_SAMPLE_EQUITY = True
PLOT_FINAL_BAL_HIST = True

# ---------------------------
# Helper functions
# ---------------------------

def _get_risk_amount(balance: float, risk_pct: float) -> float:
    """Compute risk amount for a given balance, applying risk cap if set."""
    base = math.floor(balance * risk_pct)
    if MAX_RISK_DOLLARS is not None:
        return min(base, MAX_RISK_DOLLARS)
    return base

def simulate_once(
    starting_balance: float = STARTING_BALANCE,
    num_trades: int = NUM_TRADES,
    win_prob: float = WIN_PROB,
    rrr_low: float = RRR_LOW,
    rrr_high: float = RRR_HIGH,
    risk_pct: float = RISK_PCT,
    commission: float = COMMISSION_PER_TRADE,
    slippage_pct: float = SLIPPAGE_PCT,
    calc_tax: bool = CALC_TAX,
    tax_pct: float = TAX_PCT,
    defer_tax_to_end: bool = DEFER_TAX_TO_END,
    reduce_risk_after_dd_pct: float = REDUCE_RISK_AFTER_DD_PCT,
    reduced_risk_pct: float = REDUCED_RISK_PCT,
    report_trades: bool = REPORT_TRADES,
    report_interval: int = REPORT_INTERVAL,
    rng: random.Random = None
) -> Dict:
    """Run a single simulation and return a dictionary of results and metrics."""
    if rng is None:
        rng = random.Random()

    balance = float(starting_balance)
    balance_history = [balance]
    peak_balance = balance
    max_drawdown = 0.0
    realized_profit_total = 0.0  # for deferred tax vs per-trade
    trades_info = []

    # tracking stats
    wins = 0
    losses = 0
    win_rewards = []
    loss_amounts = []
    current_losing_streak = 0
    longest_losing_streak = 0

    curr_risk_pct = risk_pct

    for i in range(num_trades):
        # adjust risk percent if drawdown threshold reached
        if reduce_risk_after_dd_pct is not None and max_drawdown >= reduce_risk_after_dd_pct:
            curr_risk_pct = min(curr_risk_pct, reduced_risk_pct)

        rrr = round(rng.uniform(rrr_low, rrr_high), 2)
        is_win = rng.random() < win_prob

        risk_amount = _get_risk_amount(balance, curr_risk_pct)
        # ensure minimum risk of $1 for making meaningful trade
        risk_amount = max(1, int(risk_amount))

        if is_win:
            gross_reward = risk_amount * rrr
            slippage_cost = gross_reward * slippage_pct
            # commission reduces reward
            net_reward_before_tax = max(0.0, gross_reward - slippage_cost - commission)
            if calc_tax and not defer_tax_to_end:
                tax = net_reward_before_tax * tax_pct / 100.0
            else:
                tax = 0.0
            net_reward = math.floor(net_reward_before_tax - tax)
            balance += net_reward
            realized_profit_total += max(0.0, net_reward) if defer_tax_to_end else 0.0
            wins += 1
            win_rewards.append(net_reward)
            current_losing_streak = 0
            outcome_amount = net_reward
            outcome_str = "Win"
        else:
            # loss includes slippage + commission
            slippage_cost = risk_amount * slippage_pct
            loss_total = math.floor(risk_amount + slippage_cost + commission)
            balance -= loss_total
            losses += 1
            loss_amounts.append(loss_total)
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
            outcome_amount = -loss_total
            outcome_str = "Loss"

        balance = max(math.floor(balance), 0)  # prevent negative balance
        balance_history.append(balance)

        # update peak and drawdown
        peak_balance = max(peak_balance, balance)
        drawdown = (peak_balance - balance) / peak_balance * 100.0 if peak_balance > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

        trades_info.append({
            "trade_index": i + 1,
            "outcome": outcome_str,
            "rrr": rrr,
            "risk_amount": risk_amount,
            "pnl": outcome_amount,
            "balance": balance
        })

        if report_trades:
            print(f"{i+1:4d} | {outcome_str:4s} | RRR {rrr:6.2f} | Risk ${risk_amount:6,} | PnL ${outcome_amount:7,} | Bal ${balance:8,}")

        if balance <= 0:
            # account depleted
            break

    # end of trades: apply deferred tax (if enabled)
    if calc_tax and defer_tax_to_end:
        net_profit = max(0.0, balance - starting_balance)
        tax_due = net_profit * tax_pct / 100.0
        balance = math.floor(balance - tax_due)

    # summary stats
    total_trades_executed = len(trades_info)
    win_rate = (wins / total_trades_executed) * 100.0 if total_trades_executed > 0 else 0.0
    total_profit = balance - starting_balance

    # expectancy: average net PnL per trade divided by average risk (expressed as percent of risk)
    pnl_series = np.array([t["pnl"] for t in trades_info], dtype=float)
    avg_pnl = pnl_series.mean() if pnl_series.size > 0 else 0.0
    avg_risk = np.mean([t["risk_amount"] for t in trades_info]) if trades_info else 1.0
    expectancy_per_trade = (avg_pnl / avg_risk) if avg_risk != 0 else 0.0

    # profit factor (sum wins / sum losses)
    sum_wins = sum([p for p in pnl_series if p > 0])
    sum_losses = -sum([p for p in pnl_series if p < 0])  # positive number
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else float('inf')

    # simple risk-adjusted metric (mean/ std of trade returns)
    # trade returns relative to risk
    trade_returns = pnl_series / np.array([t["risk_amount"] for t in trades_info], dtype=float)
    mean_return = float(np.mean(trade_returns)) if trade_returns.size > 0 else 0.0
    std_return = float(np.std(trade_returns)) if trade_returns.size > 0 else 1.0
    sharpe_like = mean_return / std_return if std_return != 0 else float('inf')

    # longest drawdown already computed
    results = {
        "final_balance": balance,
        "total_profit": total_profit,
        "win_rate_pct": win_rate,
        "wins": wins,
        "losses": losses,
        "max_drawdown_pct": max_drawdown,
        "expectancy_per_trade": expectancy_per_trade,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
        "longest_losing_streak": longest_losing_streak,
        "balance_history": balance_history,
        "trades_info": trades_info
    }

    return results


def run_monte_carlo(runs: int = NUM_MONTE_CARLO_RUNS, seed: int = SEED):
    rng = random.Random(seed)
    final_balances = []
    sample_run = None

    for i in range(runs):
        # create a new RNG for each run to avoid correlated sequences but keep reproducible
        run_seed = rng.randint(0, 2**30 - 1)
        run_rng = random.Random(run_seed)
        res = simulate_once(rng=run_rng, report_trades=False)
        final_balances.append(res["final_balance"])
        # keep the first run as a sample for plotting
        if i == 0:
            sample_run = res

    final_balances = np.array(final_balances, dtype=float)
    return final_balances, sample_run


# ---------------------------
# Main entry: run simulation(s) and display results
# ---------------------------
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    if NUM_MONTE_CARLO_RUNS <= 1:
        # single-run mode: show trade table optionally and equity curve
        res = simulate_once(report_trades=REPORT_TRADES, report_interval=REPORT_INTERVAL, rng=random.Random(SEED))
        bal_hist = res["balance_history"]

        print("\n=== Single Simulation Summary ===")
        print(f"Final Balance: ${res['final_balance']:,.0f}")
        print(f"Total Profit:  ${res['total_profit']:,.0f}")
        print(f"Wins / Losses: {res['wins']} / {res['losses']} (Win Rate: {res['win_rate_pct']:.2f}%)")
        print(f"Maximum Drawdown: {res['max_drawdown_pct']:.2f}%")
        print(f"Expectancy (per trade / relative to avg risk): {res['expectancy_per_trade']:.4f}")
        print(f"Profit Factor: {res['profit_factor']:.3f}")
        print(f"Sharpe-like (mean/std of trade returns): {res['sharpe_like']:.3f}")
        print(f"Longest losing streak: {res['longest_losing_streak']} trades")

        if PLOT_SAMPLE_EQUITY:
            plt.figure(figsize=(10, 5))
            plt.plot(bal_hist, linewidth=1.2)
            plt.title("Equity Curve (Single Simulation)")
            plt.xlabel("Trade Number")
            plt.ylabel("Account Balance ($)")
            plt.grid(True)
            plt.show()

    else:
        # Monte Carlo mode
        final_balances, sample_run = run_monte_carlo(NUM_MONTE_CARLO_RUNS, seed=SEED)

        median_final = np.median(final_balances)
        mean_final = np.mean(final_balances)
        pct_5 = np.percentile(final_balances, 5)
        pct_95 = np.percentile(final_balances, 95)
        pct_25 = np.percentile(final_balances, 25)
        pct_75 = np.percentile(final_balances, 75)
        positive_pct = np.mean(final_balances > STARTING_BALANCE) * 100.0

        print("\n=== Monte Carlo Summary ===")
        print(f"Runs: {NUM_MONTE_CARLO_RUNS}")
        print(f"Median Final Balance: ${median_final:,.0f}")
        print(f"Mean Final Balance:   ${mean_final:,.0f}")
        print(f"5th Percentile:       ${pct_5:,.0f}")
        print(f"95th Percentile:      ${pct_95:,.0f}")
        print(f"25th / 75th:          ${pct_25:,.0f} / ${pct_75:,.0f}")
        print(f"% Runs > Starting Bal (${STARTING_BALANCE:,}): {positive_pct:.2f}%")

        # show sample equity curve
        if PLOT_SAMPLE_EQUITY and sample_run is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(sample_run["balance_history"], label="Sample Run Equity")
            plt.title("Sample Run Equity Curve")
            plt.xlabel("Trade Number")
            plt.ylabel("Account Balance ($)")
            plt.grid(True)
            plt.show()

        if PLOT_FINAL_BAL_HIST:
            plt.figure(figsize=(10, 5))
            plt.hist(final_balances, bins=60)
            plt.title("Distribution of Final Balances (Monte Carlo)")
            plt.xlabel("Final Balance ($)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
