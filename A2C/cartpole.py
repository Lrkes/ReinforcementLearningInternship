import gymnasium as gym





env = gym.make("CartPole-v1")

state_size = env.observation_space
action_size = env.action_space

print(state_size)
print(action_size)