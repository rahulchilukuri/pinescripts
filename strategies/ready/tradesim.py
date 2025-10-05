import random
import numpy as np
import math

# Configurable parameters
starting_balance = 65000
num_trades = 250
report_interval = 50
seed = 25
risk_percent = 0.01

# Validate inputs
assert risk_percent > 0, "Risk percent must be positive"
assert num_trades > 0, "Number of trades must be positive"
assert report_interval > 0, "Report interval must be positive"

# Generate randomized sample trades with 25-33% win rate and RRR between 3 and 30
random.seed(seed)
np.random.seed(seed)
num_sample_trades = 10
target_wins = random.randint(2, 3)  # 20-30% win rate (2-3 wins out of 10)
example_trades = []
for i in range(num_sample_trades):
    rrr = round(random.uniform(3, 30), 2)  # Random RRR between 3 and 30
    win = i < target_wins  # First 2-3 trades are wins to achieve 20-30% win rate
    example_trades.append({"rrr": rrr, "win": win})

# Shuffle to avoid predictable win/loss order
random.shuffle(example_trades)

# Randomize trades
trades = [random.choice(example_trades) for _ in range(num_trades)]

# Simulate trading and print each trade
balance = starting_balance
balance_history = [balance]
max_balance = balance
max_drawdown = 0

# print("Trade Details:")
# print(f"{'Trade':<6} {'Outcome':<8} {'RRR':<8} {'Risk':<8} {'Reward/Loss':<12} {'Balance':<12}")
# print("-" * 50)

for i, trade in enumerate(trades):
    risk_amount = math.floor(balance * risk_percent)
    if trade["win"]:
        reward = math.floor(risk_amount * trade["rrr"])
        balance += reward
        outcome = "Win"
        amount = f"+${reward:,}"
    else:
        balance -= risk_amount
        outcome = "Loss"
        amount = f"-${risk_amount:,}"
    balance = max(math.floor(balance), 0)  # Prevent negative balance
    balance_history.append(balance)
    
    # Update max drawdown
    max_balance = max(max_balance, balance)
    drawdown = (max_balance - balance) / max_balance * 100
    max_drawdown = max(max_drawdown, drawdown)
    
    # Print trade details
    # print(f"{i+1:<6} {outcome:<8} {trade['rrr']:<8.2f} ${risk_amount:<8,} {amount:<12} ${balance:<12,}")
    
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

# Ensure final trade is reported
if len(balance_history) - 1 % report_interval != 0 and balance > 0:
    i = len(balance_history) - 1
    balance_val = balance_history[-1]
    wins_so_far = sum(1 for t in trades if t["win"])
    total_so_far = len(trades)
    win_rate = (wins_so_far / total_so_far * 100) if total_so_far > 0 else 0
    cum_return_pct = ((balance_val / starting_balance) - 1) * 100
    period_return_pct = ((balance_val / previous_balance) - 1) * 100
    print(f"Trade {i:3d}: ${balance_val:,}  |  CR: {cum_return_pct:6.2f}%  |  PR: {period_return_pct:6.2f}%  |  WR: {win_rate:6.2f}%")
