import gymnasium as gym
import torch
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("Acrobot-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
seed = 0

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)
# Load the trained model
agent.qnetwork_local.load_state_dict(torch.load('weights/dqn_checkpoint.pth', map_location=device))

# Set the agent to evaluation mode
agent.qnetwork_local.eval()
# Exploitation
agent.eps = 0.0

for i_episode in range(5):
    state = env.reset()
    state = state[0]
    done = False
    score = 0
    while not done:
        env.render()
        action = agent.act(state)
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        score += reward
    print(f"Episode {i_episode + 1} Score: {score}")

env.close()
