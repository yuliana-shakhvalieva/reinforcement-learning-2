import numpy as np
import torch
import random

from constants import SEED, device

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class BatchDQN:
    def __init__(self, batch):
        self.walls = []
        self.preys = []
        self.agents = []
        self.distances = []

        self.actions = []

        self.next_walls = []
        self.next_preys = []
        self.next_agents = []
        self.next_distances = []

        self.rewards = []
        self.flags = []

        self._parse_data(batch)
        self._tensor_convert()

    def _parse_data(self, batch):
        for value in batch:
            self.walls.append(value[0])
            self.preys.append(value[1])
            self.agents.append(value[2])
            self.distances.append(value[3])

            self.actions.append(value[4])

            self.next_walls.append(value[5])
            self.next_preys.append(value[6])
            self.next_agents.append(value[7])
            self.next_distances.append(value[8])

            self.rewards.append(value[9])
            self.flags.append(value[10])

    def _tensor_convert(self):
        self.walls = torch.from_numpy(np.array(self.walls)).float().unsqueeze(2).to(device)
        self.preys = torch.from_numpy(np.array(self.preys)).float().unsqueeze(2).to(device)
        self.agents = torch.from_numpy(np.array(self.agents)).float().unsqueeze(2).to(device)
        self.distances = torch.from_numpy(np.array(self.distances)).float().unsqueeze(2).to(device)

        self.actions = torch.from_numpy(np.array(self.actions)).int().to(device)

        self.next_walls = torch.from_numpy(np.array(self.next_walls)).float().unsqueeze(2).to(device)
        self.next_preys = torch.from_numpy(np.array(self.next_preys)).float().unsqueeze(2).to(device)
        self.next_agents = torch.from_numpy(np.array(self.next_agents)).float().unsqueeze(2).to(device)
        self.next_distances = torch.from_numpy(np.array(self.next_distances)).float().unsqueeze(2).to(device)

        self.rewards = torch.from_numpy(np.array(self.rewards)).float().to(device)
        self.flags = torch.from_numpy(np.array(self.flags)).to(device)