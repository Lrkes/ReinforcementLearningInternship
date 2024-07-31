import torch
import torch.nn.functional as f


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.softmax(self.fc2(x), dim=1)
        return x


print(f"Using {device} device")
