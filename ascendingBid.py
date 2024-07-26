import numpy as np
import matplotlib.pyplot as plt

def is_nash_equilibrium(strategy_profile, payoffs, player):
    """
    Check if a strategy profile is a Nash equilibrium for a given player.
    """
    # Get the payoff for the player at the given strategy profile
    player_payoff = payoffs[strategy_profile[0]][strategy_profile[1]][player]

    # Check if the player has a better payoff by deviating from their current strategy
    for i in range(len(payoffs)):
        if payoffs[i][strategy_profile[1]][player] > player_payoff:
            return False
    return True

# Define the strategic form matrix and strategies
payoffs = [
    [(0, 0), (0, 0)],
    [(0, 0), (0, 0)]
]
alice_strategies = ["Aggressive", "Conservative"]
bob_strategies = ["Aggressive", "Conservative"]

# Check each strategy profile to see if it's a Nash equilibrium
nash_equilibria = []
for i in range(len(payoffs)):
    for j in range(len(payoffs[0])):
        if is_nash_equilibrium((i, j), payoffs, 0) and is_nash_equilibrium((i, j), payoffs, 1):
            nash_equilibria.append((i, j))

# Convert payoffs to numpy array for heatmap plotting
payoffs_np = np.array([[payoff[0] for payoff in row] for row in payoffs])

# Print the Nash equilibria
if len(nash_equilibria) > 0:
    print("Nash Equilibrium(s):")
    for equilibrium in nash_equilibria:
        print(f"  {alice_strategies[equilibrium[0]]}, {bob_strategies[equilibrium[1]]}")
else:
    print("No Nash Equilibrium found.")

# Plot the heatmap
plt.imshow(payoffs_np, cmap='viridis', interpolation='nearest')

# Add labels and ticks
plt.title('Strategic Form Matrix')
plt.xlabel('Bob')
plt.ylabel('Alice')
plt.xticks(np.arange(len(bob_strategies)), bob_strategies)
plt.yticks(np.arange(len(alice_strategies)), alice_strategies)

# Highlight Nash equilibria
for equilibrium in nash_equilibria:
    plt.scatter(equilibrium[1], equilibrium[0], color='red', marker='o', s=200, label='Nash Equilibrium')

plt.legend()

# Show plot
plt.colorbar(label='Alice\'s Payoff')
plt.show()
