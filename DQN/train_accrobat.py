import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dqn_agent import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)

n_episodes = 1000
max_t = 200

scores = []
scores_window = deque(maxlen=50)
average_scores = []
epsilons = []

for i_episode in range(1, n_episodes + 1):
    state = env.reset()[0]
    score = 0

    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    # Record metrics
    scores.append(score)
    scores_window.append(score)
    average_score = np.mean(scores_window)
    average_scores.append(average_score)
    epsilons.append(agent.eps)
    agent.update_epsilon()

    print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.2f}", end="")
    if i_episode % 50 == 0:
        print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.2f}")

torch.save(agent.qnetwork_local.state_dict(), 'weights/dqn_checkpoint.pth')

# Plots
plt.figure(figsize=(15, 10))

# Plot raw scores
plt.subplot(3, 1, 1)
plt.plot(scores, label='Score')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Scores Over Time')

# Plot average scores
plt.subplot(3, 1, 2)
plt.plot(average_scores, label='Average Score', color='orange')
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Average Score Over Last 50 Episodes')

# Plot epsilon decay
plt.subplot(3, 1, 3)
plt.plot(epsilons, label='Epsilon', color='green')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay Over Time')

plt.tight_layout()
plt.savefig('Visualization/accrobat/training_metrics.png')
plt.show()

env.close()
