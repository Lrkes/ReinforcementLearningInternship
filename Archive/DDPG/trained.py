import gymnasium as gym
from notWorkingDDPG.agent import DDPGAgent

env = gym.make('Pendulum-v1', render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = DDPGAgent(state_size=state_size, action_size=action_size, size=64)
print(f"Input file Files Score:")
# input = input()
agent.actor.load_weights('-1328actor.weights.h5')
agent.critic.load_weights('-1328critic.weights.h5')

num_runs = 3
max_steps = 200

for run in range(num_runs):
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step += 1
        env.render()
    print(f"Run {run + 1}: Total Reward: {total_reward}")

env.close()
