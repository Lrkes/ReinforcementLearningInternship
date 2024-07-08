import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

# Explicitly set device to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('Acrobot-v1')

state_size = env.observation_space.shape[0]
print(state_size)
action_size = env.action_space.n
seed = 0

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)

n_episodes = 1000
max_t = 200
scores = []
scores_window = []

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
    agent.update_epsilon()
    print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.2f}", end="")
    if i_episode % 100 == 0:
        print(f"\rEpisode {i_episode}\tAverage Score: {average_score:.2f}")

# Save the trained model
torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/dqn_checkpoint.pth')

# Plotting the scores
plt.figure(figsize=(10,5))
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Scores Over Time')
plt.savefig('training_scores.png')
plt.show()

env.close()
