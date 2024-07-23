import gymnasium as gym
import torch
import numpy as np
from dqn_agent import DQNAgent
from model import QNetwork

# Explicitly set device to CPU
device = torch.device("cpu")

env = gym.make("MountainCar-v0", render_mode="human")

# Define settings
learning_rate = 0.01
batch_size = 64
update_every = 10
network_size = 64
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
gamma = 0.9

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 0

agent = DQNAgent(state_size=state_size, action_size=action_size, gamma=gamma, seed=seed, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, update_every=update_every, size=network_size, lr=learning_rate, batch_size=batch_size)

# Load the trained model
agent.qnetwork_local.load_state_dict(torch.load('checkpoints/mCar_checkpoint.pth', map_location=device))

# Set the agent to evaluation mode
agent.qnetwork_local.eval()

# Exploitation
agent.eps = 0.0

for i_episode in range(5):
    state = env.reset()
    state = state[0]
    done = False
    score = 0
    while not done:
        env.render()
        action = agent.act(state)
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        score += reward
    print(f"Episode {i_episode + 1} Score: {score}")

env.close()
