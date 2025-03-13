import numpy as np
import pickle
import random
from taxi_env import FullTaxiEnv

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
epsilon_min = 0.01  # Minimum exploration rate
max_steps = 1000  # Set a step threshold per episode
num_episodes = 5000  # Number of training episodes

# Initialize Q-table
q_table = {}

def get_q_value(state, action):
    """
    Returns the Q-value for a given state-action pair, initializing if necessary.
    Available Actions: 
        • 0: Move South (Down)
        • 1: Move North (Up)
        • 2: Move East  (Right)
        • 3: Move West  (Left)
        • 4: Pick up passenger (PICKUP) 
        • 5: Drop off passenger (DROPOFF)
    """
    if state not in q_table:
        q_table[state] = np.zeros(6)  # 6 possible actions
    return q_table[state][action]

def update_q_value(state, action, reward, next_state):
    """Updates the Q-table using the Q-learning update rule."""
    if state not in q_table:
        q_table[state] = np.zeros(6)
    if next_state not in q_table:
        q_table[next_state] = np.zeros(6)
    
    best_next_action = np.argmax(q_table[next_state])
    q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

def choose_action(state):
    """Chooses an action using an epsilon-greedy policy."""
    if state not in q_table:
        q_table[state] = np.zeros(6)
    if np.random.rand() < epsilon:
        return random.choice([0, 1, 2, 3, 4, 5])  # Random action
    else:
        return np.argmax(q_table[state]) # Greedy action

# Training loop
for episode in range(num_episodes):
    env = FullTaxiEnv()
    state, _ = env.reset()
    done = False
    rewards_per_episode = []
    total_reward = 0
    episode_steps = 0
    
    while not done and episode_steps < max_steps:
        episode_steps += 1

        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Reward Shaping
        
        
        
        update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    
    rewards_per_episode.append(total_reward)
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {epsilon:.4f}")
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"🚀 Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

# Save trained Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete with reward shaping. Q-table saved as q_table.pkl.")
