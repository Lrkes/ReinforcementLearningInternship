import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

def main():
    env = gym.make("Taxi-v3", render_mode='rgb_array')
    
    # Initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0  # exploration vs. exploitation 1.0 => 100 % for random action
    decay_rate = 0.007

    # Training variables
    num_episodes = 1000
    max_steps = 100


    # For Plots and prints
    scores = []
    epsilon_values = []


    # Training
    for episode in range(num_episodes):
        # Reset the environment
        state, _ = env.reset()
        done = False
        total_rewards = 0 
        

        for s in range(max_steps):
            # Behavior policy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # Array with Actions that have the highest Q-Value for the State
                max_indices = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
                # Select a Random Action from max_indices
                action = np.random.choice(max_indices)

            # Take action and observe reward
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-learning algorithm (Target policy)
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            total_rewards += reward

            if done:
                break

        # For Visualization
        scores.append(total_rewards)
        epsilon_values.append(epsilon)

        # Decrease epsilon
        epsilon = max(0.01, np.exp(-decay_rate * episode))


    # Results plotted
    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Plot 1: Epsilon Over Time
    axes[0].plot(epsilon_values, label='Epsilon Value')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Epsilon')
    axes[0].set_title('Epsilon Value Over Time')
    axes[0].legend()

    # Plot 2: Score vs. Episode
    axes[1].plot(scores, label='Score')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Score vs Episode')
    axes[1].legend()

    # Plot 3: Learning Curve with Rolling Average
    window_size = 50
    rolling_avg_rewards = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    axes[2].plot(rolling_avg_rewards, label='Rolling Average Reward')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Average Reward')
    axes[2].set_title('Learning Curve')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('Visualization/taxi/results.png')
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
