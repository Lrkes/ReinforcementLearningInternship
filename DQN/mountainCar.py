import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

# Explicitly set device to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("MountainCar-v0")

# Define settings
learning_rate = 0.001
batch_size = 64
update_every = 5
network_size = 64
buffer_size = 125000
eps_start = 1
eps_end = 0.01
eps_decay = 0.995
gamma = 0.96

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 0

agent = DQNAgent(state_size=state_size, action_size=action_size, gamma=gamma, seed=seed, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, buffer_size=buffer_size, update_every=update_every, size=network_size, lr=learning_rate, batch_size=batch_size)

n_episodes = 1000
max_t = 200
scores = []
scores_window = []
average_scores = []
epsilons = []

for i_episode in range(1, n_episodes + 1):
    state = env.reset()
    state = state[0]
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
torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/mCar_checkpoint.pth')

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

# Add settings to the plot
plt.figtext(0.15, 0.01, f'Learning Rate: {learning_rate}\nBatch Size: {batch_size}\nUpdate Every: {update_every}\nBuffer Size: {buffer_size}\ngamma: {gamma}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()

# Save the plot with a unique filename
plt.savefig(f'Visualization/mCar/training_metrics_lr{learning_rate}_bs{batch_size}_update{update_every}_buffer{buffer_size}.png')
plt.show()

env.close()
