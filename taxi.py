import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt


def main():

    # create Taxi environment
    env = gym.make("Taxi-v3")

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0 # exploration vs. exploitation 1.0 => 100 % for random action
    decay_rate = 0.006

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    scores = []
    non_negative_rewards = 0

    scores_per_10 = []

    # training
    for episode in range(num_episodes):
        
        # reset the environment
        state, _ = env.reset()
        done = False
        total_rewards = 0

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # take action and observe reward
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            total_rewards += reward

            
            # if done, finish episode
            if done:
                break

        # Print Q-values after each episode
        print(f"Q-values for episode {episode}:")
        print(qtable[state, :])
        print()

        scores.append(total_rewards)
        scores_per_10.append(total_rewards)
        
        if episode % 10 == 0:
            average = np.mean(scores_per_10)
            print(f"the current episode is {episode} and the epsilon value is {epsilon}. Average reward for the last 10 episodes is: {average}")
            scores_per_10 = []

        if total_rewards >= 0:
            non_negative_rewards += 1

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")

    print(f"Average reward: {np.mean(scores)}")
    print(f"non negative rewards: {non_negative_rewards}")

    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score vs Episode')
    plt.show()
    
    input("Press Enter to watch trained agent...")



    # watch trained agent
    state, _ = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))
        action = np.argmax(qtable[state, :])
        new_state, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    main()
