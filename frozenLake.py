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
    state_visits = np.zeros(state_size)

    learning_rate = 0.825
    discount_rate = 0.9
    epsilon = 1.0
    decay_rate = 0.005

    num_episodes = 5000
    max_steps = 100
    
    heatmap_interval = 250
    num_heatmaps = num_episodes // heatmap_interval

    all_state_visits = np.zeros((num_heatmaps, state_size))

    action_labels = ['←', '↓', '→', '↑']
    rewards_per_episode = []
    win_count_per_interval = []
    current_win_count = 0

    q_values = [[] for _ in range(state_size)]  # Track Q-values for all states
    all_q_values = [[] for _ in range(state_size)]


    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        state_visits[state] += 1
        total_reward = 0

        for step in range(max_steps):



            if random.uniform(0, 1) < epsilon or np.all(qtable[state, :] == qtable[state, 0]):
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            qtable[state, action] += learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            state_visits[new_state] += 1
            state = new_state
            total_reward += reward
                
            if episode < 1000:  
                # Correct nested loops to track Q-values for all states
                for s in range(state_size):  # Using 's' to avoid conflict with the outer loop's 'state'
                    for action in range(action_size):
                        all_q_values[s].append(qtable[s, action])

            if done:
                break
            

        rewards_per_episode.append(total_reward)
        
        if total_reward > 0:
            current_win_count += 1
        
        epsilon = np.exp(-decay_rate * episode)

        if not (episode + 1) % heatmap_interval:
            all_state_visits[episode // heatmap_interval] = state_visits
            win_count_per_interval.append(current_win_count * 100 / heatmap_interval)
            current_win_count, state_visits = 0, np.zeros(state_size)

    env.close()

    print(f"Total rewards: {sum(rewards_per_episode)}")

    best_actions = [np.argmax(qtable[state, :]) for state in range(state_size)]
    actions = [
        'x' if state in [5, 7, 11, 12] else ('o' if state == 15 else action_labels[best_actions[state]])
        for state in range(state_size)
    ]

    best_action_grid = np.array(actions).reshape((4, 4))

    # 1. Heatmaps with State Visits per 250 Episodes
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i >= num_heatmaps:
            ax.axis('off')
            continue
        sns.heatmap(all_state_visits[i].reshape(4, 4), annot=True, cmap="YlGnBu", fmt=".0f", ax=ax)
        for (x, y), color in [((0, 3), "red"), ((1, 1), "red"), ((3, 1), "red"), ((3, 2), "red"), ((3, 3), "green")]:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=color, lw=2))
        ax.set_title(f"Episodes {i * heatmap_interval}-{i * heatmap_interval + heatmap_interval - 1}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=best_action_grid, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.show()

    # 2. Shows Reward over time
    episodes_per_split = 1000
    splits = num_episodes // episodes_per_split

    fig, axes = plt.subplots(nrows=splits, ncols=1, figsize=(12, 6 * splits))
    for i in range(splits): 
        start = i * episodes_per_split
        end = (i + 1) * episodes_per_split
        ax = axes[i]
        ax.plot(range(start, end), rewards_per_episode[start:end])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Total Reward per Episode (Episodes {start + 1} to {end})")
    plt.tight_layout()
    plt.show()

    # 3. Plot the win percentage per 250 episodes
    plt.figure(figsize=(12, 6))
    plt.plot(range(heatmap_interval, num_episodes + 1, heatmap_interval), win_count_per_interval, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Win Percentage")
    plt.title("Win Percentage per 250 Episodes")
    plt.show()


     # 4. Q-Value over time
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))
    
    for state in range(state_size):
        row = state // 4
        col = state % 4
        for action in range(action_size):
            axes[row, col].plot(all_q_values[state][action::state_size], label=f'Action {action_labels[action]}')
        axes[row, col].set_xlabel('Steps', fontsize=10)
        axes[row, col].set_ylabel('Q-value', fontsize=10)
        axes[row, col].set_title(f'State {state}', fontsize=12)
        axes[row, col].legend(fontsize=8)


    plt.suptitle('Q-values for each Action in each State Over 250 Episodes', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    main()
