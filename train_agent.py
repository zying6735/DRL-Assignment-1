import numpy as np
import pickle
import random
from taxi_env import FullTaxiEnv


def update_q_value(state, action, reward, next_state):
    state_q = state[10:]
    next_state_q = next_state[10:]
    
    if state_q not in q_table:
        q_table[state_q] = np.zeros(6)
    if next_state_q not in q_table:
        q_table[next_state_q] = np.zeros(6)

    best_next_action = np.argmax(q_table[next_state_q])
    q_table[state_q][action] += alpha * (reward + gamma * q_table[next_state_q][best_next_action] - q_table[state_q][action])


def choose_action(state):
    state_q = state[10:]
    if state_q not in q_table:
        return random.choice(range(6))
    return random.choice(range(6)) if np.random.rand() < epsilon else np.argmax(q_table[state_q])

"""
Training loop
"""
# Initialize Q-table
q_table = {}
# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.01
max_steps = 10000
num_episodes = 100000

for episode in range(num_episodes):
    env = FullTaxiEnv()
    state, _ = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0

    stations = [(state[i], state[i + 1]) for i in range(2, 10, 2)]
    visited_stations = set()
    curr_passenger_on_taxi = False
    destination_loc = None
    passenger_loc = None

    while not done and episode_steps < max_steps:
        episode_steps += 1
        
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)

        taxi_pos_curr = (next_state[0], next_state[1])
        obstacle_north_curr, obstacle_south_curr, obstacle_east_curr, obstacle_west_curr = next_state[10:14]
        passenger_look_curr, destination_look_curr = next_state[14:16]

        taxi_north_curr = (taxi_pos_curr[0] - 1, taxi_pos_curr[1])
        taxi_south_curr = (taxi_pos_curr[0] + 1, taxi_pos_curr[1])
        taxi_east_curr = (taxi_pos_curr[0], taxi_pos_curr[1] + 1)
        taxi_west_curr = (taxi_pos_curr[0], taxi_pos_curr[1] - 1)

        shaped_reward = 0.0

        if not curr_passenger_on_taxi:
            if action not in {0,1,2,3,4}:
                shaped_reward -= 10
            for station in stations:
                if station in [taxi_north_curr, taxi_south_curr, taxi_east_curr, taxi_west_curr]:
                    if station not in visited_stations:
                        shaped_reward += 10
                        if passenger_look_curr:
                            passenger_loc = station
                            visited_stations.add(station)
                            shaped_reward += 10
                        elif destination_look_curr:
                            destination_loc = station
                            visited_stations.add(station)
                            shaped_reward += 10
                        else:
                            visited_stations.add(station)

        if curr_passenger_on_taxi:
            if action == 5 and destination_look_curr == 0:
                shaped_reward -= 10
            for station in stations:
                if station in [taxi_north_curr, taxi_south_curr, taxi_east_curr, taxi_west_curr]:
                    if station not in visited_stations:
                        shaped_reward += 10
                        if destination_look_curr:
                            destination_loc = station
                            shaped_reward += 10
                        else:
                            visited_stations.add(station)
            
        if taxi_pos_curr == passenger_loc and action == 4:
            curr_passenger_on_taxi = True
            shaped_reward += 10

        if curr_passenger_on_taxi and taxi_pos_curr == destination_loc and action == 5:
            curr_passenger_on_taxi = False
            shaped_reward += 50

        if curr_passenger_on_taxi:
            if action == 5 and taxi_pos_curr != destination_loc:
                curr_passenger_on_taxi = False
                shaped_reward -= 10

        if (obstacle_north_curr == 1 or obstacle_south_curr == 1) and action in {0, 1}:
            shaped_reward -= 5
        if (obstacle_east_curr == 1 or obstacle_west_curr == 1) and action in {2, 3}:
            shaped_reward -= 5

        if episode_steps >= 50:
            shaped_reward -= 1

        shaped_reward -= 0.1

        update_q_value(state, action, reward + shaped_reward, next_state)
        state = next_state
        total_reward += reward + shaped_reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"🚀 Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# Save trained Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as q_table.pkl.")
