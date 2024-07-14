from collections import deque
import random

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.memory.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

# Example usage
buffer_size = 24
batch_size = 8

# Initialize replay memory
replay_memory = ReplayMemory(buffer_size)

# Fill replay memory with random experiences
for i in range(buffer_size):
    experience = random.randint(0, 999)  # Example experience (random number)
    print(f"{i}: {experience}")
    replay_memory.add_experience(experience)


# Sample a batch of experiences from replay memory
batch = replay_memory.sample_batch(batch_size)
print("Sampled Batch:", batch)


