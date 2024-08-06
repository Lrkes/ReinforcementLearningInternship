import gymnasium as gym
import reinforce as r
import torch.optim as optim


env = gym.make("CartPole-v1")

state_size = env.observation_space
action_size = env.action_space


print(state_size)
print(action_size)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = r.PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
