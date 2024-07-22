import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
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
scores_window = []
average_scores = []
epsilons = []

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

    # Scores
    scores.append(score)
    scores_window.append(score)
    if len(scores_window) > 100:
        scores_window.pop(0)
    average_score = np.mean(scores_window)
    average_scores.append(average_score)
    epsilons.append(agent.eps)
    agent.update_epsilon()
    print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.2f}", end="")
    if i_episode % 100 == 0:
        print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.2f}")

# Save the trained model
torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/dqn_checkpoint.pth')

# Plotting the scores
plt.figure(figsize=(15, 10))

# Plot the raw scores
plt.subplot(3, 1, 1)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Scores Over Time')

# Plot the average scores
plt.subplot(3, 1, 2)
plt.plot(np.arange(1, len(average_scores) + 1), average_scores)
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Average Score Over Last 100 Episodes')

# Plot the epsilon decay
plt.subplot(3, 1, 3)
plt.plot(np.arange(1, len(epsilons) + 1), epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay Over Time')

plt.tight_layout()
plt.savefig('Visualization/accrobat/training_metrics.png')
plt.show()

env.close()
