import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
import math


class FullTaxiEnv():
    def __init__(self, fuel_limit=50):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = random.randint(5, 10)  # randomly assign grid size

        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        """
        Generate 4 random positions for Passenger and Destination
        """
        self.stations = []
        while len(self.stations) < 4:
             pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
             if pos not in self.stations:
                 self.stations.append(pos)
            
        self.passenger_loc = None
        self.destination = None

        """
        Generate position of obstacles
        """
        total_cells = self.grid_size * self.grid_size
        obstacle_density = random.randint(0, 30) / 100  # Random density (0% to 30%)
        # calculate the number of the obstacle
        num_obstacles = math.ceil(total_cells * obstacle_density)
        
        self.obstacles = []
        while len(self.obstacles) < num_obstacles:
            pos_obstacle = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos_obstacle not in self.obstacles and pos_obstacle not in self.stations:
                self.obstacles.append(pos_obstacle)
        
    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        
        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]

        self.taxi_pos = random.choice(available_positions)
        
        self.passenger_loc = random.choice([pos for pos in self.stations])
        
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
        return self.get_state(), {}

    def step(self, action):
        """
        Perform an action and update the environment state.
         Return: state, reward, done, info
         Available Actions: 
            • 0: Move South (Down)
            • 1: Move North (Up)
            • 2: Move East  (Right)
            • 3: Move West  (Left)
            • 4: Pick up passenger (PICKUP) 
            • 5: Drop off passenger (DROPOFF)
        """
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col

        reward = 0

        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5  # Run into obstacle
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    # Incorrect Pickup
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        # Successfully Deliver
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        # Incorrect Dropoff
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    # Incorrect Dropoff
                    reward -=10
        
        reward -= 0.1  # Efficiency Penalty
        self.current_fuel -= 1  # Fuel Comsumption (-1  each step)

        if self.current_fuel <= 0:
            # Feul Management Penalty and Terminate
            return self.get_state(), reward -10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    
    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['⬜'] * self.grid_size for _ in range(self.grid_size)]
        
        """
        grid[0][0]='R'
        grid[0][4]='G'
        grid[4][0]='Y'
        grid[4][4]='B'
        """
        
        # Place passenger
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = '🤖'
        
        # Place destination
        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = '🏢'
        
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Place obstacles
        for oy, ox in self.obstacles:
            grid[oy][ox] = '🟫'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = FullTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        action = student_agent.get_action(obs)
        print(action)
        obs, reward, done, _ = env.step(action)
        print('obs:',obs)
        total_reward += reward
        step_count += 1
        print("action reward: ", reward)

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True) 
    print(f"Final Score: {agent_score}")
