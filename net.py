import torch.nn as nn
import numpy as np
import random
import torch

from constants import SEED, NUM_ACTIONS, DEVICE, NUM_AGENTS

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(5, 32, kernel_size=3, stride=1),
                                   nn.ELU(),
                                   nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                   nn.ELU(),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                   nn.ELU(),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=2),
                                   nn.ELU(),
                                   nn.Flatten(),
                                   nn.Linear(256, 128),
                                   nn.ELU(),
                                   nn.Linear(128, 64),
                                   nn.ELU(),
                                   nn.Linear(64, NUM_ACTIONS)).to(DEVICE)

    def forward(self, agent_states):
        return self.model(agent_states)

    def get_q_values(self, agents_states):
        q_values = []
        for i in range(NUM_AGENTS):
            q_values.append(self.forward(agents_states[:, i, :, :, :]))

        return torch.stack(q_values, dim=1)
