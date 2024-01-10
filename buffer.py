from collections import namedtuple, deque
import torch
import numpy as np
import random

from constants import SEED

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

Transition = namedtuple('Transition',
                        ('agents_states', 'action', 'next_agents_states', 'reward', 'flag'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)