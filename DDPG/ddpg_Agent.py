import os
import keras
from keras import layers
import tensorflow as tf
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

num_states = 3
num_actions = 1
upper_bound = 2
lower_bound = -2


# This OUActionNoise class adds noise to the action output of the actor model to encourage exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch


class DDPGAgent:
    def __init__(self, critic_lr=0.002, actor_lr=0.001):
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)

        self.buffer = Buffer(50000, 64)
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

        self.gamma = 0.99
        self.tau = 0.005

    def get_actor(self):
        # Initializing weights between -3e-3 and 3-e3 ensures stable initial actions, prevents saturation,
        # and promotes smooth learning
        action_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        # initializations of the output layer with small values because they directly output the action
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=action_init)(out)
        outputs = outputs * upper_bound

        model = tf.keras.Model(inputs, outputs)
        return model

    # def get_actor(self):
    #     # Initializing weights between -3e-3 and 3-e3 ensures stable initial actions, prevents saturation,
    #     # and promotes smooth learning
    #     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    #     inputs = layers.Input(shape=(num_states,))
    #     out = layers.Dense(128, activation="relu")(inputs)
    #     out = layers.Dense(128, activation="relu")(out)
    #     # initializations of the output layer with small values because they directly output the action
    #     outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    #     outputs = outputs * upper_bound
    #     model = tf.keras.Model(inputs, outputs)
    #     return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(num_states,))
        state_out = layers.Dense(32, activation="relu")(state_input)

        # Action as input
        action_input = layers.Input(shape=(num_actions,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Combining the state and action paths
        concat = layers.Concatenate()([state_out, action_out])
        concat_out = layers.Dense(32, activation="relu")(concat)

        output = layers.Dense(1)(concat_out)

        model = tf.keras.Model([state_input, action_input], output)

        return model

    # def get_critic(self):
    #     state_input = layers.Input(shape=(num_states,))
    #     state_out = layers.Dense(16, activation="relu")(state_input)
    #     state_out = layers.Dense(32, activation="relu")(state_out)
    #
    #     action_input = layers.Input(shape=(num_actions,))
    #     action_out = layers.Dense(32, activation="relu")(action_input)
    #
    #     concat = layers.Concatenate()([state_out, action_out])
    #     out = layers.Dense(256, activation="relu")(concat)
    #     out = layers.Dense(256, activation="relu")(out)
    #     outputs = layers.Dense(1)(out)
    #     model = tf.keras.Model([state_input, action_input], outputs)
    #     return model

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Use automatic differentiation to calculate gradients for optimizing the models
        # The GradientTape records the operations for the forward pass
        # The chain rule is applied during the backward pass to compute gradients

        # Update critic model
        with tf.GradientTape() as tape:
            # Compute target actions using the target actor model
            target_actions = self.target_actor(next_state_batch, training=True)
            # Compute target Q-values: reward + discounted target Q-value from target critic model
            target_q = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            # Compute predicted Q-values using the critic model with the current state and action
            q_value = self.critic_model([state_batch, action_batch], training=True)
            # Compute the loss between target Q-values and predicted Q-values
            critic_loss = tf.math.reduce_mean(tf.math.square(target_q - q_value))

        # Compute gradients of the critic loss with respect to critic model's variables
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # Apply the gradients to update the critic model's weights
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Update actor model
        with tf.GradientTape() as tape:
            # Compute actions using the actor model with the current state
            actions = self.actor_model(state_batch, training=True)
            # Compute the Q-value of these actions using the critic model
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Compute the actor loss: negative mean of the Q-values (actor aims to maximize Q-value)
            actor_loss = -tf.math.reduce_mean(critic_value)

        # Compute gradients of the actor loss with respect to actor model's variables
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        # Apply the gradients to update the actor model's weights
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    # Samples the buffer for the update method
    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()
        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def update_target(self, target_weights, weights, tau):
        """
        Update the target network weights using soft update.

        Parameters:
        - target_weights: List or iterable of target network variables (weights).
        - weights: List or iterable of current network variables (weights).
        - tau: Float, the update rate for the target network (0 < tau < 1).
        """
        for target, current in zip(target_weights, weights):
            # Perform the soft update: target = tau * current + (1 - tau) * target
            target.assign(current * tau + target * (1 - tau))

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        sampled_actions = sampled_actions.numpy() + noise
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
        return [np.squeeze(legal_action)]
