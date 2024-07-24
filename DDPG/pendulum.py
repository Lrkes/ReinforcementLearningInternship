import gymnasium as gym
import numpy as np
from ddpg_agent import DDPGAgent
from collections import deque

env = gym.make('Pendulum-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = DDPGAgent(state_size=state_size, action_size=action_size, size=64)

episodes = 500
max_steps = 200
scores = deque(maxlen=50)  # To store the last 50 episode scores

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    agent.noise.reset()
    score = 0
    step = 0
    done = False
    
    while not done and step < max_steps:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        step += 1
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

    scores.append(score)
    average_score = np.mean(scores)
    
    print(f"\rEpisode {episode}\tAverage Score (last 50 episodes): {average_score:.2f}", end="")

    if episode % 50 == 0:
        print(f"\rEpisode {episode}\tAverage Score (last 50 episodes): {average_score:.2f}")

env.close()
