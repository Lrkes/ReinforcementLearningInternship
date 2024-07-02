import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)

    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # State visit counter (reshaped for heatmap)
    state_visits = np.zeros(state_size)

    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    num_episodes = 10000
    max_steps = 100
    heatmap_interval = 100
    num_heatmaps = num_episodes // heatmap_interval
    
    all_state_visits = np.zeros((num_heatmaps, state_size))

    best_actions = []
    action_labels = ['←', '↓', '→', '↑']
    actions = []
    
    # Initialize a list to store the rewards
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        # Initial state
        state_visits[state] += 1

        total_reward = 0  # Initialize total reward for the episode

        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon or np.all(qtable[state, :] == qtable[state, 0]):
                action = env.action_space.sample()  # Choose random action
            else:
                action = np.argmax(qtable[state, :])  # Choose the action with the highest Q-value

            new_state, reward, terminated, truncated, info = env.step(action)
            # Increment the visit counter for the current state
            done = terminated or truncated

            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            state_visits[new_state] += 1

            state = new_state

            total_reward += reward  # Update total reward

            if done:
                break

        rewards_per_episode.append(total_reward)  # Store the total reward for this episode

        epsilon = np.exp(-decay_rate * episode)
        
        if (episode + 1) % heatmap_interval == 0:
            heatmap_index = episode // heatmap_interval
            all_state_visits[heatmap_index] = state_visits
            if episode != num_episodes - 1:  # Don't reset for the last episode
                state_visits = np.zeros(state_size)

    env.close()

    for state in range(state_size):
        best_action = np.argmax(qtable[state, :])
        best_actions.append(best_action)

        # Apply state-based logic
        if state in [5, 7, 11, 12]:
            actions.append('x')  # Mark as 'x'
        elif state == 15:
            actions.append('o')  # Mark as 'o'
        else:
            actions.append(action_labels[best_action])
    actions = np.array(actions)

    best_action_grid = actions.reshape((4, 4))

    # Display all heatmaps
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(all_state_visits[i].reshape(4, 4), annot=True, cmap="YlGnBu", fmt=".0f", ax=ax)
        # Mark specific cells (example coordinates)
        for (x, y), color in [((0, 3), "red"), ((1, 1), "red"), ((3, 1), "red"), ((3, 2), "red"), ((3, 3), "green")]:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=color, lw=2))  # Add box
        ax.set_title(f"Episodes {i * heatmap_interval}-{i * heatmap_interval + heatmap_interval - 1}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    plt.tight_layout()

    # Create figure and axes
    fig, ax = plt.subplots()

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create table
    table = ax.table(cellText=best_action_grid, loc='center', cellLoc='center')

    # Adjust table properties (optional)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust cell size

    # Show plot
    plt.show()

    # Plot the reward over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.show()

if __name__ == "__main__":
    main()
