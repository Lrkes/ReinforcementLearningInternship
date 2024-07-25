import gymnasium as gym
import numpy as np
from ddpg_agent import DDPGAgent
from collections import deque
import datetime

env = gym.make('Pendulum-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = DDPGAgent(state_size=state_size, action_size=action_size, size=128)

# agent.actor.save_weights('actor_weights.h5')
# agent.critic.save_weights('critic_weights.h5')


episodes = 5000
max_steps = 200
scores = deque(maxlen=50)

final_scores = scores

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

final_scores = round(average_score)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the trained models
agent.actor.save_weights(f'{final_scores}actor.weights.h5')
agent.critic.save_weights(f'{final_scores}critic.weights.h5')

print(current_time)

env.close()