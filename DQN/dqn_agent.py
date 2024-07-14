import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, lr=0.001, tau=0.005, update_every=4):

        # Environment info and seed
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Neural networks and optimizer
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)  # Local network for training
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)  # Target network for stable Q-value estimation
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)  # Optimizer for training the local network
        
        # Experience replay memory
        self.memory = deque(maxlen=buffer_size)  # Replay buffer to store experiences
        self.batch_size = batch_size  # Batch size for sampling from replay buffer
        
        # Discount factor for future rewards
        self.gamma = gamma
        
        # Soft update parameter for target network
        self.tau = tau
        
        # Update frequency and time step counter
        self.update_every = update_every  # Frequency of learning step
        self.t_step = 0  # Counter to track the number of steps
        
        # Exploration-exploitation parameters
        self.eps = 1.0  # Initial epsilon value for exploration
        self.eps_end = 0.01  # Minimum epsilon value
        self.eps_decay = 0.995  # Decay rate for epsilon


    # Method that determines what the agent will do next.
    def step(self, state, action, reward, next_state, done):
        # Add Info to Memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Updates the Step Counter
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                # Sample experiences if the step reaches 0 and the memorz is larger than the batch size
                experiences = self.sample()
                # Learn on the previously sampled data
                self.learn(experiences, self.gamma)
    
    def act(self, state):
        # Convert the state to a tensor and add a batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Set the network to evaluation mode (important for accurate predictions)
        self.qnetwork_local.eval()

        # Get action values from the local network without updating the network
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Set the network back to training mode (to resume learning later)
        self.qnetwork_local.train()
        

        # TODO: Change like FrozenLake?
        # Choose the action with the highest value
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Otherwise, explore by selecting a random action
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        # Unpack the batch of experiences
        states, actions, rewards, next_states, dones = experiences

        # Get the maximum predicted Q-values for the next states from the target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the Q-targets for the current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get the expected Q-values from the local model for the actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute the loss (mean squared error between expected and target Q-values)
        loss = F.mse_loss(Q_expected, Q_targets)
        # Zero the gradients
        self.optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Update the weights
        self.optimizer.step()

        # Soft update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        # Soft update model parameters:
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample(self):
        # Randomly sample a batch of experiences from memory
        experiences = random.sample(self.memory, self.batch_size)
    
        # Convert the batch of experiences to PyTorch tensors
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # Return the batch of experiences as tuples
        return (states, actions, rewards, next_states, dones)

    def update_epsilon(self):
        # Decay the epsilon value for exploration-exploitation trade-off
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
