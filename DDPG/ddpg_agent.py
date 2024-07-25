import numpy as np
import tensorflow as tf
from collections import deque
import random
from network import Actor, Critic

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_deviation = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_size, action_size, size=128, lr_actor=0.0005, lr_critic=0.001, gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_size, action_size, size)
        self.critic = Critic(state_size, action_size, size)
        self.target_actor = Actor(state_size, action_size, size)
        self.target_critic = Critic(state_size, action_size, size)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.noise = OUActionNoise(mean=np.zeros(action_size), std_deviation=float(0.2) * np.ones(action_size))

    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * tau + target * (1.0 - tau))

    @tf.function
    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)
        action += tf.convert_to_tensor(self.noise(), dtype=tf.float32)
        action = tf.clip_by_value(action, -1.0, 1.0)
        return action[0]

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        if len(self.buffer) > self.batch_size:
            self.learn()

    @tf.function
    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Critic update
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions)
        target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values = self.critic(states, actions)
            critic_loss = tf.math.reduce_mean(tf.math.square(target_q_values - q_values))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.math.reduce_mean(self.critic(states, actions_pred))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Update target networks
        self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic.variables, self.tau)

        # Log additional information
        #tf.print("Critic Loss:", critic_loss, "Actor Loss:", actor_loss)
        #tf.print("Sample Actions:", actions)
        #tf.print("Predicted Actions:", actions_pred)
        #tf.print("Sample Q-values:", q_values)
        # tf.print("Target Q-values:", target_q_values)
