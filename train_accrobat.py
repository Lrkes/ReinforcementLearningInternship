import gymnasium as gym
import torch
import numpy as np
from dqn_agent import DQNAgent

# Explicitly set device to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('Acrobot-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 0

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)

n_episodes = 1000
max_t = 200
scores = []

for i_episode in range(1, n_episodes + 1):
    state = env.reset()
    state = state[0]  # Acrobot-v1 returns a tuple with state and additional info
    score = 0
    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores.append(score)
    agent.update_epsilon()
    print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}", end="")
    if i_episode % 100 == 0:
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}")

# Save the trained model
torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoint.pth')

env.close()
