import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class SmallDQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(SmallDQN, self).__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
in_states = 4  # Example input states (e.g., in FrozenLake-v1 environment)
h1_nodes = 16  # Example hidden layer nodes
out_actions = 4  # Example output actions (e.g., left, down, right, up in FrozenLake)

small_dqn = SmallDQN(in_states, h1_nodes, out_actions)

# Create a dummy input tensor for visualization (4 input states)
dummy_input = torch.randn(1, in_states)

# Generate a graph of the neural network architecture
graph = make_dot(small_dqn(dummy_input), params=dict(small_dqn.named_parameters()))

# Save the graph to a file or display it
graph.render("small_dqn_graph", format="png")  # Save as PNG file
graph.view()  # Display in default viewer (requires Graphviz installed)
