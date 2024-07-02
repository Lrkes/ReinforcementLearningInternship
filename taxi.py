import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

def main():
    # create Taxi environment with rgb_array render mode for recording
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

    scores = []
    non_negative_rewards = 0
    scores_per_10 = []

    # Variables for Plot 1: Exploration vs. Exploitation
    exploration_count = 0
    exploitation_count = 0

    exploration_percentage = []

    q_values = [[] for _ in range(action_size)]  # Liste von leeren Listen für jeden Q-Wert



    # def training():
    # Training
    for episode in range(num_episodes):

        # Wrap environment with RecordVideo for every 100th episode
        # if episode % 100 == 0 or episode == num_episodes - 1:
        #    env = RecordVideo(env, video_folder='videos', episode_trigger=lambda x: True, name_prefix=episode)

        # Reset the environment
        state, _ = env.reset()
        done = False
        total_rewards = 0 
        

        for s in range(max_steps):

            # Exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # Explore
                action = env.action_space.sample()
                exploration_count += 1
            else:
                # Exploit
                action = np.argmax(qtable[state, :])
                exploitation_count += 1

            # Take action and observe reward
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            total_rewards += reward


            # If done, finish episode
            if done:
                break

        # QValues

        scores.append(total_rewards)
        scores_per_10.append(total_rewards)

        for action in range(action_size):
            q_values[action].append(qtable[3, action])



        if episode % 10 == 0:
            average = np.mean(scores_per_10)
            print(f"The current episode is {episode} and the epsilon value is {epsilon}. Average reward for the last 10 episodes is: {average}")
            scores_per_10 = []

            exploration_pct = (exploration_count / (exploration_count + exploitation_count)) * 100
            exploration_percentage.append(exploration_pct)

        if total_rewards >= 0:
            non_negative_rewards += 1

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

        # Close the wrapped environment to save the video and revert to the base environment
        if episode % 100 == 0:
            env.close()
            env = gym.make("Taxi-v3", render_mode='rgb_array')

        
    
    print(f"Training completed over {num_episodes} episodes")
    print(f"Non-negative rewards: {non_negative_rewards}")

    print("Total exploration steps:", exploration_count)
    print("Total exploitation steps:", exploitation_count)


    # Create a plot for each position in the arrays

    # Plot all lines on the same graph
    for action in range(action_size):
        plt.plot(q_values[action], label=f'Action {action}')

    # Add labels and a legend
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.title('Q-Values for Each Action')
    plt.legend()

    # Show the plot
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot 1: Exploration vs. Exploitation
    axes[0].plot(exploration_percentage)
    axes[0].set_xlabel('Episode (in intervals of 10)')
    axes[0].set_ylabel('Exploration Percentage')
    axes[0].set_title('Exploration vs. Exploitation Over Time')

    # Plot 2: Score vs. Episode
    axes[1].plot(scores)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Score vs Episode')

    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
