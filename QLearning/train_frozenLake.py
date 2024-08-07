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

    # Hyperparameter
    learning_rate = 0.825
    discount_rate = 0.9
    epsilon = 1.0
    decay_rate = 0.005

    # Episodes
    num_episodes = 5000
    max_steps = 100

    # For plots
    state_visits = np.zeros(state_size)
    plot_interval = 250
    num_heatmaps = num_episodes // plot_interval
    all_state_visits = np.zeros((num_heatmaps, state_size))

    win_count_per_interval = []
    current_win_count = 0

    all_q_values = [[] for _ in range(state_size)]

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        state_visits[state] += 1
        total_reward = 0

        for step in range(max_steps):

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # Array with Actions that have the highest Q-Value for the State
                max_indices = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
                # Select a Random Action from max_indices
                action = np.random.choice(max_indices)

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Value updates: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
            qtable[state, action] += learning_rate * (
                    reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            state_visits[new_state] += 1

            state = new_state

            total_reward += reward

            if episode < 1000:
                for s in range(state_size):
                    for action in range(action_size):
                        all_q_values[s].append(qtable[s, action])

            if done:
                break

        if total_reward > 0:
            current_win_count += 1

        epsilon = np.exp(-decay_rate * episode)

        if not (episode + 1) % plot_interval:
            all_state_visits[episode // plot_interval] = state_visits
            win_count_per_interval.append(current_win_count * 100 / plot_interval)
            current_win_count, state_visits = 0, np.zeros(state_size)

    env.close()

    # Plotting results

    # 1. Heatmaps with State Visits per 250 Episodes
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i >= num_heatmaps:
            ax.axis('off')
            continue
        sns.heatmap(all_state_visits[i].reshape(4, 4), annot=True, cmap="YlGnBu", fmt=".0f", ax=ax)
        for (x, y), color in [((0, 3), "red"), ((1, 1), "red"), ((3, 1), "red"), ((3, 2), "red"), ((3, 3), "green")]:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=color, lw=2))
        ax.set_title(f"Episodes {i * plot_interval}-{i * plot_interval + plot_interval - 1}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    plt.tight_layout()
    plt.savefig('Visualization/frozenLake/heatmap_statevisits2.png')
    plt.show()

    # 2. Best actions Map
    action_labels = ['←', '↓', '→', '↑']
    best_actions = [np.argmax(qtable[state, :]) for state in range(state_size)]
    actions = [
        'x' if state in [5, 7, 11, 12] else ('o' if state == 15 else action_labels[best_actions[state]])
        for state in range(state_size)
    ]

    best_action_grid = np.array(actions).reshape((4, 4))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=best_action_grid, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.savefig('Visualization/frozenLake/best_actions2.png')
    plt.show()

    # 3. Plot the win percentage per 250 episodes
    plt.figure(figsize=(12, 6))
    plt.plot(range(plot_interval, num_episodes + 1, plot_interval), win_count_per_interval, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Win Percentage")
    plt.title("Win Percentage per 250 Episodes")
    plt.savefig('Visualization/frozenLake/win_percentage2.png')
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
    plt.savefig('Visualization/frozenLake/Q-Values2.png')
    plt.show()


if __name__ == "__main__":
    main()
