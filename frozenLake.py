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

    num_episodes = 1000
    max_steps = 100
    heatmap_interval = 100
    num_heatmaps = num_episodes // heatmap_interval
    
    all_state_visits = np.zeros((num_heatmaps, state_size))




    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)
            # Increment the visit counter for the current state
            state_visits[new_state] += 1
            done = terminated or truncated

            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            
            
            env.render()

            state = new_state

            if done:
                break

        epsilon = np.exp(-decay_rate * episode)
        
        if (episode + 1) % heatmap_interval == 0:
            heatmap_index = episode // heatmap_interval
            all_state_visits[heatmap_index] = state_visits
            if episode != num_episodes - 1:  # Don't reset for the last episode
                state_visits = np.zeros(state_size) 

    env.close()
    #Display all heatmaps
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(all_state_visits[i].reshape(4, 4), annot=True, cmap="YlGnBu", fmt=".0f", ax=ax)
        # Mark specific cells (example coordinates)
        for (x, y), color in [(0, 3),  "red"], [(1, 1), "red"], [(3, 1), "red"], [(3, 2), "red"], [(3, 3), "green"]:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=color, lw=2))  # Add box
        ax.set_title(f"Episodes {i * heatmap_interval}-{i * heatmap_interval + heatmap_interval - 1}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    plt.tight_layout()
    plt.show()
    

    

if __name__ == "__main__":
    main()



