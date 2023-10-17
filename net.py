from torch import nn
import torch

from constants import SEED, FIELS_SIDE, NUM_ACTIONS, device

torch.manual_seed(SEED)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.walls_head = nn.Sequential(nn.Conv2d(FIELS_SIDE, 256, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 64, kernel_size=3, padding=1)).to(device)

        self.preys_head = nn.Sequential(nn.Conv2d(FIELS_SIDE, 256, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 64, kernel_size=3, padding=1)).to(device)
        
        self.common_part = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 4, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.Flatten(),
                                         nn.Linear(160, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 16),
                                         nn.ReLU(),
                                         nn.Linear(16, NUM_ACTIONS)).to(device)

    def forward(self, walls, preys):
        conv_walls = self.walls_head(walls)
        conv_preys = self.preys_head(preys)

        return self.common_part((conv_walls + conv_preys))
