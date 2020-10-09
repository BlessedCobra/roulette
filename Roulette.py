import gym
import numpy as np
import random
from IPython.display import clear_output

"""
This is a simulated casino Roulette environment from 1-36, including a 0,
where the agent can bet either even (actions 0, 2, 4, ..., 34),
odd (actions 1, 3, 5, ..., 35), 0 (action 36), or
walk away without betting anything (action 37)

The reward for correctly betting either even or odd is +1, and -1 for incorrect betting.
The reward for correctly betting 0 is +35, and -1 for incorrect betting.
The reward for walking away is 0.

As you can see from the final value of the q-table, the agent quickly learns that the best
result is to walk away: any other action results in a net negative reward,
just like at real casinos.
"""

env = gym.make('Roulette-v0')

# initialize variables
step = 0
steps_total = []
episodes = 100000

# Q-table initialization
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
# Note: These values are arbitrary
alpha = 0.1
gamma = 0.7
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

# Loop every episode
for i in range(episodes):
    # Reset at start of every episode
    state = env.reset()
    reward = 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Exploration
        else:
            action = np.argmax(q_table[state]) # Exploitation

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Bellman Equation and update q_table
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # iterate to next state
        state = next_state
        
    # Output every 10000 episodes
    if i % 10000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

print(q_table)

env.close()