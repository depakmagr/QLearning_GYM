import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
 
# Grid parameters
rows, cols = 10,10
n_states = rows * cols
goal=(rows-1,cols-1)
n_actions = 4  # up, right, down, left
 
# Hyperparameters
alpha = 0.01
gamma = 0.9
epsilon = 0.1
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
    if action == 0: r = max(r - 1, 0)        # up
    elif action == 1: c = min(c + 1, cols-1) # right
    elif action == 2: r = min(r + 1, rows-1) # down
    elif action == 3: c = max(c - 1, 0)      # left
    return (r, c)
 
# State embedding (one-hot)
def state_to_tensor(state):
    idx = state_to_index(state)
    one_hot = np.zeros(n_states)
    one_hot[idx] = 1.0
    return torch.FloatTensor(one_hot).unsqueeze(0)
 
# DQN model
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
 
    def forward(self, x):
        return self.fc(x)
 
# Initialize model
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()
 
# Training loop
for episode in range(episodes):
    state = (0, 0)
    while not is_terminal(state):
        s_tensor = state_to_tensor(state)
 
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            with torch.no_grad():
                q_values = model(s_tensor)
                action = torch.argmax(q_values).item()
 
        next_state = get_next_state(state, action)
        reward = 10 if is_terminal(next_state) else -1
 
        s_next_tensor = state_to_tensor(next_state)
        with torch.no_grad():
            target = reward + gamma * model(s_next_tensor).max().item()
 
        q_pred = model(s_tensor)[0, action]
        loss = loss_fn(q_pred, torch.tensor(target))
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        state = next_state
 
# Show final policy
actions = ["↑", "→", "↓", "←"]
policy_grid = []
 
for i in range(rows):
    row = []
    for j in range(cols):
        state = (i, j)
        if is_terminal(state):
            row.append("🏁")
        else:
            with torch.no_grad():
                q_vals = model(state_to_tensor(state))
                row.append(actions[torch.argmax(q_vals).item()])
    policy_grid.append(row)
 
for row in policy_grid:
    print(" ".join(row))