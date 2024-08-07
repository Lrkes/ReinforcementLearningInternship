import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt


def create_actor(state_shape, action_shape, action_bound):
    inputs = layers.Input(shape=state_shape)
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(action_shape[0], activation="tanh")(out)

    outputs = outputs * action_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def create_critic(state_shape, action_shape):
    state_input = layers.Input(shape=state_shape)
    action_input = layers.Input(shape=action_shape)

    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(400, activation="relu")(concat)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model


class DDPG:
    def __init__(self, state_shape, action_shape, action_bound, gamma=0.99, tau=0.005, actor_lr=0.001, critic_lr=0.002,
                 update_every=10000):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau

        self.actor = create_actor(state_shape, action_shape, action_bound)
        self.critic = create_critic(state_shape, action_shape)

        self.target_actor = create_actor(state_shape, action_shape, action_bound)
        self.target_critic = create_critic(state_shape, action_shape)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.update_target(self.target_actor.variables, self.actor.variables, tau=1.0)
        self.update_target(self.target_critic.variables, self.critic.variables, tau=1.0)

        self.buffer = []
        self.buffer_limit = 1000000
        self.batch_size = 64

    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * tau + target * (1 - tau))

    def get_action(self, state, noise_scale):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0]
        action += noise_scale * np.random.randn(self.action_shape[0])
        return np.clip(action, -self.action_bound, self.action_bound)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_limit:
            self.buffer.pop(0)

    def sample_batch(self):
        indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        return states, actions, rewards, next_states, dones

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(tf.convert_to_tensor(next_states, dtype=tf.float32), training=True)
            y = rewards + self.gamma * (1 - dones) * self.target_critic(
                [tf.convert_to_tensor(next_states, dtype=tf.float32), target_actions], training=True)
            critic_value = self.critic(
                [tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(actions, dtype=tf.float32)],
                training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(tf.convert_to_tensor(states, dtype=tf.float32), training=True)
            critic_value = self.critic([tf.convert_to_tensor(states, dtype=tf.float32), actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic.variables, self.tau)


env = gym.make('MountainCarContinuous-v0')
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_bound = env.action_space.high[0]

agent = DDPG(state_shape, action_shape, action_bound)

num_episodes = 200
noise_scale = 1.0
noise_decay = 0.995
min_noise_scale = 0.01
rewards = []

for episode in range(num_episodes):
    state = env.reset()[0]
    episode_reward = 0

    while True:
        action = agent.get_action(state, noise_scale)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        episode_reward += reward

        if done:
            break

    rewards.append(episode_reward)
    noise_scale = max(min_noise_scale, noise_scale * noise_decay)

    print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Noise: {noise_scale:.2f}")

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
