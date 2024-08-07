import tensorflow as tf
import gymnasium as gym
from ddpg_agent import DDPGAgent

env = gym.make("Pendulum-v1", render_mode="human")

agent = DDPGAgent()

agent.actor_model.load_weights("weights/pendulum_actor.weights.h5")
agent.critic_model.load_weights("weights/pendulum_critic.weights.h5")
agent.target_actor.load_weights("weights/pendulum_target_actor.weights.h5")
agent.target_critic.load_weights("weights/pendulum_target_critic.weights.h5")

for ep in range(3):
    env.reset()
    episodic_reward = 0
    done = False
    truncated = False

    while not done and not truncated:
        env.render()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = agent.policy(tf_prev_state)  # No noise added
        state, reward, done, truncated, _ = env.step(action)
        episodic_reward += reward
        prev_state = state

    print(f"Episode {ep + 1} Reward: {episodic_reward}")

env.close()
