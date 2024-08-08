import tensorflow as tf
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent

env = gym.make("Pendulum-v1")
agent = DDPGAgent()

total_episodes = 250

ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = agent.policy(tf_prev_state)
        state, reward, done, truncated, _ = env.step(action)

        agent.buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        agent.learn()
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables, agent.tau)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables, agent.tau)

        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, round(avg_reward, 2)))
    avg_reward_list.append(avg_reward)

plt.figure(figsize=(15, 10))
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.savefig('Visualization/pendulum/training_metrics.png')
plt.show()



agent.actor_model.save_weights("weights/actor.weights.h5")
agent.critic_model.save_weights("weights/critic.weights.h5")
agent.target_actor.save_weights("weights/target_actor.weights.h5")
agent.target_critic.save_weights("weights/target_critic.weights.h5")
