from torch import nn
import torch

from constants import SEED, FIELS_SIDE, NUM_ACTIONS, device

torch.manual_seed(SEED)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.walls_head = nn.Sequential(nn.Conv2d(FIELS_SIDE, 64, kernel_size=3, padding=1)).to(device)

        self.preys_head = nn.Sequential(nn.Conv2d(FIELS_SIDE, 64, kernel_size=3, padding=1)).to(device)

        self.agents_head = nn.Sequential(nn.Conv2d(FIELS_SIDE, 64, kernel_size=3, padding=1)).to(device)

        self.distances_head = nn.Sequential(nn.Conv2d(FIELS_SIDE, 64, kernel_size=3, padding=1)).to(device)

        self.common_part = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.Flatten(),
                                         nn.Linear(640, NUM_ACTIONS)).to(device)

    def forward(self, walls, preys, agents, distances):
        conv_walls = self.walls_head(walls)
        conv_preys = self.preys_head(preys)
        conv_agents = self.agents_head(agents)
        conv_distances = self.distances_head(distances)

        return self.common_part((conv_walls + conv_preys + conv_agents + conv_distances))
