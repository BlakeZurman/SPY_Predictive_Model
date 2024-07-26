import numpy as np
import matplotlib.pyplot as plt

# Initialize players' strategies
# 'A' represents aggressive strategy, 'C' represents conservative strategy
# You can adjust the risk factor for each player
# Higher risk factor makes the player more likely to bid aggressively
player_A_risk_factor = 0.8  # Adjust this value for Player A
player_B_risk_factor = 0.008  # Adjust this value for Player B

# Initialize bidding history
bids_A = []
bids_B = []

# Start the auction
current_bid = 0
for round in range(10):
    print("Round", round+1)
    
    # Player A's bid strategy based on Player B's previous bid
    if round > 0:
        if bids_B[round - 1] > bids_A[round - 1]:
            player_A_strategy = 'A' if np.random.rand() < player_A_risk_factor else 'C'
        else:
            player_A_strategy = 'A' if np.random.rand() < player_A_risk_factor else 'C'
    else:
        player_A_strategy = 'A' if np.random.rand() < player_A_risk_factor else 'C'
    
    # Player B's bid strategy based on Player A's previous bid
    if round > 0:
        if bids_A[round - 1] > bids_B[round - 1]:
            player_B_strategy = 'A' if np.random.rand() < player_B_risk_factor else 'C'
        else:
            player_B_strategy = 'A' if np.random.rand() < player_B_risk_factor else 'C'
    else:
        player_B_strategy = 'A' if np.random.rand() < player_B_risk_factor else 'C'
    
    # Player A's bid
    if player_A_strategy == 'A':
        bid_A = current_bid + np.random.randint(1, 500)
    else:
        bid_A = current_bid + np.random.randint(1, 100)
    
    # Player B's bid
    if player_B_strategy == 'A':
        bid_B = current_bid + np.random.randint(1, 500)
    else:
        bid_B = current_bid + np.random.randint(1, 100)
    
    # Update current bid
    current_bid = max(bid_A, bid_B)
    
    # Record bids
    bids_A.append(bid_A)
    bids_B.append(bid_B)
    
    print("Player A bid:", bid_A)
    print("Player B bid:", bid_B)
    print("Current winning bid:", current_bid)
    print()

# Plotting the progression of bids
rounds = range(1, 11)
plt.plot(rounds, bids_A, label='Player A')
plt.plot(rounds, bids_B, label='Player B')
plt.xlabel('Round')
plt.ylabel('Bid')
plt.title('Progression of Bids in Car Auction')
plt.legend()
plt.show()
