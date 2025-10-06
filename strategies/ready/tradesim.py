import numpy as np
import math

# Configurable parameters
starting_balance = 10000
num_trades = 300
report_interval = 30
seed = 23
calc_tax = True
tax_pct = 30
risk_percent = 0.01
report_trades = not True
min_win_rate = 2
max_win_rate = 3
min_rr = 3
max_rr = 30

# Generate realistic sample trades
num_sample_trades = 100

# Validate inputs
assert risk_percent > 0, "Risk percent must be positive"
assert num_trades > 0, "Number of trades must be positive"
assert report_interval > 0, "Report interval must be positive"

# Set seeds for reproducibility
# np.random.seed(seed)

# Calculate target wins based on 25-30% win rate of sample trades
target_wins = np.random.randint(int(num_sample_trades * 0.25), int(num_sample_trades * 0.30))

example_trades = []

for i in range(num_sample_trades):
    win = i < target_wins

    if win:
        # Generate win RRR from a lognormal distribution
        # Use mean and sigma values chosen to fit roughly between min_rr and max_rr
        mean_rrr = (min_rr + max_rr) / 2
        # Lognormal mean in log-space
        mu = np.log(mean_rrr) - 0.5 * 0.5 ** 2  # Adjust mu so median ≈ mean_rrr
        sigma = 0.5  # Controls spread/skewness

        rrr = np.random.lognormal(mean=mu, sigma=sigma)
        # Clamp the RRR to min_rr and max_rr
        rrr = min(max_rr, max(min_rr, rrr))
    else:
        # Loss trades: random RRR between 1 and 1.25, uniform distribution
        rrr = np.random.uniform(1, 1.25)

    example_trades.append({"rrr": round(rrr, 2), "win": win})

# Shuffle example trades to randomize order
np.random.shuffle(example_trades)

# Randomize trades for the simulation from example trades
trades = [np.random.choice(example_trades) for _ in range(num_trades)]

# Simulate trading
balance = starting_balance
balance_history = [balance]
max_balance = balance
max_drawdown = 0

if report_trades:
    print("Trade Details:")
    print(f"{'Trade':<6} {'Outcome':<8} {'RRR':<8} {'Risk':<8} {'Reward/Loss':<12} {'Balance':<12}")
    print("-" * 50)

reward = 0
for i, trade in enumerate(trades):
    risk_amount = math.floor(balance * risk_percent)

    if trade["win"]:
        if not calc_tax:
            tax_pct = 0

        profit = math.floor(risk_amount * trade["rrr"])
        tax = profit * (tax_pct / 100)
        reward = profit - tax
        balance += reward
        outcome = "Win"
        amount = f"+${reward:,.0f}"
    else:
        balance -= risk_amount
        outcome = "Loss"
        amount = f"-${risk_amount:,}"
        reward = -risk_amount

    balance = max(math.floor(balance), 0)  # Prevent negative balance
    balance_history.append(balance)

    # Update max drawdown
    max_balance = max(max_balance, balance)
    drawdown = (max_balance - balance) / max_balance * 100
    max_drawdown = max(max_drawdown, drawdown)

    if report_trades and reward > 0:
        print(f"{i+1:<6} {outcome:<8} {trade['rrr']:<8.2f} ${risk_amount:<8,} {amount:<12} ${balance:<12,}")

    if balance <= 0:
        print(f"Account depleted after {i+1} trades")
        break

print("-" * 50)

# Output summary results
final_balance = balance
total_profit = final_balance - starting_balance
wins = sum(1 for t in trades[:len(balance_history)-1] if t["win"])
win_rate = (wins / (len(balance_history)-1)) * 100 if balance > 0 else 0

print(f"\nSummary:")
print(f"Final Balance: ${final_balance:,}")
print(f"Total Profit: ${total_profit:,}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")
print(f"\nEvery {report_interval}th balance with cumulative return, period return, and win rate:")

previous_balance = starting_balance
for i in range(0, len(balance_history), report_interval):
    if i == 0:
        print(f"Trade {i:3d}: ${balance_history[i]:,}  |  CR: {0.00:6.2f}%  |  PR:    N/A  |  WR:    N/A")
        continue

    balance_val = balance_history[i]
    wins_so_far = sum(1 for t in trades[:i] if t["win"])
    total_so_far = i
    win_rate = (wins_so_far / total_so_far * 100) if total_so_far > 0 else 0
    cum_return_pct = ((balance_val / starting_balance) - 1) * 100
    period_return_pct = ((balance_val / previous_balance) - 1) * 100

    print(f"Trade {i:3d}: ${balance_val:,}  |  CR: {cum_return_pct:6.2f}%  |  PR: {period_return_pct:6.2f}%  |  WR: {win_rate:6.2f}%")
    previous_balance = balance_val

# Ensure final trade is reported if not aligned with report interval
if (len(balance_history) - 1) % report_interval != 0 and balance > 0:
    i = len(balance_history) - 1
    balance_val = balance_history[-1]
    wins_so_far = sum(1 for t in trades if t["win"])
    total_so_far = len(trades)
    win_rate = (wins_so_far / total_so_far * 100) if total_so_far > 0 else 0
    cum_return_pct = ((balance_val / starting_balance) - 1) * 100
    period_return_pct = ((balance_val / previous_balance) - 1) * 100
    print(f"Trade {i:3d}: ${balance_val:,}  |  CR: {cum_return_pct:6.2f}%  |  PR: {period_return_pct:6.2f}%  |  WR: {win_rate:6.2f}%")
