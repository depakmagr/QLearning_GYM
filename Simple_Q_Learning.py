import numpy as np
import random

#Grid parameters
rows, cols = 4, 4
n_states = rows * cols
goal=(rows-1,cols-1)
n_actions = 4 # up, right, down, left

# Q-table
Q = np.zeros((n_states, n_actions))

# Learning parameters
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.1 # exploration rate
episodes = 100

# Helper functions
def state_to_index(state):
    return state[0] * cols + state[1]

def index_to_state(index):
    return divmod(index, cols)

def is_terminal(state):
    return state == goal

def get_next_state(state, action):
    r, c = state
    if action == 0: r = max(r - 1, 0)# Up
    elif action == 1: c = min(c + 1, cols-1)# Right
    elif action == 2: r = min(r + 1, rows-1)# Down
    elif action == 3: c = max(c - 1, 0)# Left
    return(r, c)

# Training loop 
for episode in range(episodes):
    state = (0, 0)
    while not is_terminal(state):
        state_idx = state_to_index(state)

        #Epsilon-greedy action
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state_idx])
        
        next_state = get_next_state(state, action)
        next_idx = state_to_index(next_state)

        reward = 10 if is_terminal(next_state) else -1

        # Q-learning update
        Q[state_idx, action] = Q[state_idx, action] + alpha * (
            reward + gamma * np.max(Q[next_idx]) - Q[state_idx, action]
        )

        state = next_state

# Final Policy
actions = ["↑", "→", "↓", "←"]
policy_grid = []

for i in range(rows):
    row = []
    for j in range(cols):
        idx = state_to_index((i, j))
        if (i, j) == goal:
            row.append("🏴‍☠️")
        else:
            row.append(actions[np.argmax(Q[idx])])
    policy_grid.append(row)


for row in policy_grid:
    print(" ".join(row))