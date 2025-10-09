import numpy as np
import math

# ------------------------------
# CONFIGURATION
# ------------------------------
CONFIG = {
    "starting_balance": 10000,
    "num_trades": 250,
    "report_interval": 40,
    "seed": 23,
    "calc_tax": True,
    "tax_pct": 30,
    "kelly_cap": 0.16,  # Maximum 33% of full Kelly fraction
    "report_trades": not True,
    "min_rr": 3,
    "max_rr": 30,
    "num_sample_trades": 100,
    "win_rate_range": (0.24, 0.30),
    "risk_decay_factors": [1.0, 0.9, 0.8, 0.7]  # For each quarter
}

# ------------------------------
# KELLY CRITERION CALCULATION
# ------------------------------
def calculate_kelly_fraction(trades):
    # Calculate win rate (p) and average RRR for wins (b)
    wins = [t for t in trades if t["win"]]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_rrr = np.mean([t["rrr"] for t in wins]) if wins else 1

    # Kelly Criterion: f = (p * (b + 1) - 1) / b
    if avg_rrr <= 0 or win_rate <= 0:
        return 0.01  # Fallback to 1% risk if calculation is invalid
    kelly_fraction = (win_rate * (avg_rrr + 1) - 1) / avg_rrr
    return max(0, min(kelly_fraction, 1))  # Ensure fraction is between 0 and 1

# ------------------------------
# TRADE GENERATION
# ------------------------------
def generate_sample_trades(config):
    np.random.seed(config["seed"])

    min_rr = config["min_rr"]
    max_rr = config["max_rr"]
    win_rate_low, win_rate_high = config["win_rate_range"]
    num_samples = config["num_sample_trades"]
    target_wins = np.random.randint(int(num_samples * win_rate_low), int(num_samples * win_rate_high))

    trades = []
    for i in range(num_samples):
        win = i < target_wins
        if win:
            mean_rrr = (min_rr + max_rr) / 2
            mu = np.log(mean_rrr) - 0.5 * 0.5 ** 2
            sigma = 0.5
            rrr = np.random.lognormal(mean=mu, sigma=sigma)
            rrr = min(max_rr, max(min_rr, rrr))
        else:
            rrr = np.random.uniform(1, 1.25)

        trades.append({"rrr": round(rrr, 2), "win": win})

    np.random.shuffle(trades)
    return trades

# ------------------------------
# RISK DECAY STRATEGY
# ------------------------------
def adjust_risk_percent(base_risk, trade_index, total_trades, decay_factors):
    quarter = total_trades // 4
    if trade_index >= 3 * quarter:
        factor = decay_factors[3]
    elif trade_index >= 2 * quarter:
        factor = decay_factors[2]
    elif trade_index >= quarter:
        factor = decay_factors[1]
    else:
        factor = decay_factors[0]
    return base_risk * factor

# ------------------------------
# TRADE SIMULATION
# ------------------------------
def simulate_trades(config, trades):
    balance = config["starting_balance"]
    balance_history = [balance]
    max_balance = balance
    max_drawdown = 0
    reward = 0

    # Calculate Kelly fraction from sample trades
    kelly_fraction = calculate_kelly_fraction(trades)
    capped_kelly_fraction = kelly_fraction * config["kelly_cap"]  # Cap at 33% Kelly

    sampled_trades = [np.random.choice(trades) for _ in range(config["num_trades"])]

    if config["report_trades"]:
        print("Trade Details:")
        print(f"{'Trade':<6} {'Outcome':<8} {'RRR':<8} {'AR %':<8} {'Risk':<8} {'Reward/Loss':<12} {'Start Balance':<12} {'End Balance':<12}")
        print("-" * 80)

    for i, trade in enumerate(sampled_trades):
        current_risk_percent = adjust_risk_percent(capped_kelly_fraction, i, config["num_trades"], config["risk_decay_factors"])
        risk_amount = math.floor(balance * current_risk_percent)
        start = balance
        actual_risk_pcnt = (risk_amount/start)*100

        if trade["win"]:
            profit = math.floor(risk_amount * trade["rrr"])
            tax = profit * (config["tax_pct"] / 100) if config["calc_tax"] else 0
            reward = profit - tax
            balance += reward
            outcome = "Win"
            amount = f"+${reward:,.0f}"
        else:
            balance -= risk_amount
            reward = -risk_amount
            outcome = "Loss"
            amount = f"-${risk_amount:,}"

        balance = max(math.floor(balance), 0)
        balance_history.append(balance)

        max_balance = max(max_balance, balance)
        drawdown = (max_balance - balance) / max_balance * 100
        max_drawdown = max(max_drawdown, drawdown)

        if config["report_trades"]:
            print(f"{i+1:<6} {outcome:<8} {trade['rrr']:<8.2f} {actual_risk_pcnt:<8.2f} ${risk_amount:<8,} {amount:<12} ${start:<12,} ${balance:<12,}")

        if balance <= 0:
            print(f"Account depleted after {i+1} trades")
            break

    return balance_history, sampled_trades, max_drawdown

# ------------------------------
# REPORT SUMMARY
# ------------------------------
def print_trade_summary(config, balance_history, trades, max_drawdown):
    starting_balance = config["starting_balance"]
    final_balance = balance_history[-1]
    total_profit = final_balance - starting_balance
    trades_executed = len(balance_history) - 1
    wins = sum(1 for t in trades[:trades_executed] if t["win"])
    win_rate = (wins / trades_executed) * 100 if trades_executed > 0 else 0

    print("-" * 80)
    print("\nSummary:")
    print(f"Trades Executed: {trades_executed:}")
    print(f"Trades Won: {wins:}")
    print(f"Final Balance: ${final_balance:,}")
    print(f"Total Profit: ${total_profit:,}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"\nEvery {config['report_interval']}th balance with cumulative return, period return, and win rate:")

    previous_balance = starting_balance
    for i in range(0, len(balance_history), config["report_interval"]):
        balance_val = balance_history[i]
        wins_so_far = sum(1 for t in trades[:i] if t["win"])
        win_rate_i = (wins_so_far / i * 100) if i > 0 else None
        cum_return_pct = ((balance_val / starting_balance) - 1) * 100
        period_return_pct = ((balance_val / previous_balance) - 1) * 100 if i > 0 else None

        if i == 0:
            print(f"Trade {i:3d}: ${balance_val:,}  |  CR: {0.00:6.2f}%  |  PR:    N/A  |  WR:    N/A")
        else:
            print(f"Trade {i:3d}: ${balance_val:,}  |  CR: {cum_return_pct:6.2f}%  |  PR: {period_return_pct:6.2f}%  |  WR: {win_rate_i:6.2f}%")
        previous_balance = balance_val

    # Final row if not aligned
    if (len(balance_history) - 1) % config["report_interval"] != 0 and final_balance > 0:
        i = len(balance_history) - 1
        balance_val = final_balance
        wins_so_far = sum(1 for t in trades if t["win"])
        win_rate_i = (wins_so_far / len(trades) * 100)
        cum_return_pct = ((balance_val / starting_balance) - 1) * 100
        period_return_pct = ((balance_val / previous_balance) - 1) * 100
        print(f"Trade {i:3d}: ${balance_val:,}  |  CR: {cum_return_pct:6.2f}%  |  PR: {period_return_pct:6.2f}%  |  WR: {win_rate_i:6.2f}%")

# ------------------------------
# MAIN EXECUTION
# ------------------------------
def main():
    sample_trades = generate_sample_trades(CONFIG)
    balance_history, executed_trades, max_drawdown = simulate_trades(CONFIG, sample_trades)
    print_trade_summary(CONFIG, balance_history, executed_trades, max_drawdown)

if __name__ == "__main__":
    main()
