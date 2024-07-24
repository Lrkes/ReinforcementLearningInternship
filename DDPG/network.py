import tensorflow as tf
from keras import layers

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, size):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(size, activation='relu')
        self.fc2 = layers.Dense(size, activation='relu')
        self.fc3 = layers.Dense(action_size, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)


class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size, size):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(size, activation='relu')
        self.fc2 = layers.Dense(size, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, state, action):
        x = self.fc1(state)
        x = tf.concat([x, action], axis=-1)
        x = self.fc2(x)
        return self.fc3(x)
