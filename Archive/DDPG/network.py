import tensorflow as tf
from keras import layers

upper_bound = 2

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, size):
        super(Actor, self).__init__()
        self.input_layer = layers.Input(shape=(state_size,))
        self.fc1 = layers.Dense(size, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(size, activation='relu', kernel_initializer='he_normal')
        self.fc3 = layers.Dense(action_size, activation='tanh', kernel_initializer='glorot_normal')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)

class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size, size):
        super(Critic, self).__init__()
        self.fc1_state = layers.Dense(size, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(size, activation='relu', kernel_initializer='he_normal')
        self.fc3 = layers.Dense(1, kernel_initializer='glorot_normal')

    def call(self, state, action):
        state_value = self.fc1_state(state)
        x = tf.concat([state_value, action], axis=-1)
        x = self.fc2(x)
        return self.fc3(x)

def create_actor(state_size, action_size, size):
    state_input = layers.Input(shape=(state_size,))
    x = layers.Dense(size, activation='relu', kernel_initializer='he_normal')(state_input)
    x = layers.Dense(size, activation='relu', kernel_initializer='he_normal')(x)
    output = layers.Dense(action_size, activation='tanh', kernel_initializer='glorot_normal')(x)
    output = output * upper_bound
    model = tf.keras.Model(inputs=state_input, outputs=output)
    return model

def create_critic(state_size, action_size, size):
    state_input = layers.Input(shape=(state_size,))
    action_input = layers.Input(shape=(action_size,))
    
    # Define the state pathway
    state_value = layers.Dense(size, activation='relu', kernel_initializer='he_normal')(state_input)
    
    # Concatenate state and action inputs
    concat = layers.Concatenate()([state_value, action_input])
    
    # Define the action pathway
    x = layers.Dense(size, activation='relu', kernel_initializer='he_normal')(concat)
    output = layers.Dense(1, kernel_initializer='glorot_normal')(x)
    
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
    return model
