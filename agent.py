import numpy as np
import torch
from torch import nn
import random
import copy
from collections import deque
from net import Net
from batch import BatchDQN

from constants import SEED, BUFFER_MAXLEN, LEARNING_RATE, BATCH_SIZE, GAMMA, STEPS_PER_UPDATE, STEPS_PER_TARGET_UPDATE, device

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class Agent:
    def __init__(self):
        self.steps = 0
        self.buffer = deque(maxlen=BUFFER_MAXLEN)

        # Main network
        self.model = Net()

        # Target network
        self.target_model = copy.deepcopy(self.model)

        # Loss and optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.loss = nn.CrossEntropyLoss()

    def consume_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        batch_array = []
        for _ in range(BATCH_SIZE):
            idx = random.randint(0, len(self.buffer) - 1)
            batch_array.append(self.buffer[idx])
        return BatchDQN(batch_array)

    def train_step(self, batch):
        non_final_mask = torch.tensor(
            tuple(
                map(lambda s: s == False, batch.flags)
            ),
            device=device,
            dtype=torch.bool)

        non_final_next_walls = torch.stack(
            [s for s, f in zip(batch.next_walls, batch.flags.unsqueeze(1)) if f == False]
        ).to(device)

        non_final_next_preys = torch.stack(
            [s for s, f in zip(batch.next_preys, batch.flags.unsqueeze(1)) if f == False]
        ).to(device)

        non_final_next_agents = torch.stack(
            [s for s, f in zip(batch.next_agents, batch.flags.unsqueeze(1)) if f == False]
        ).to(device)

        non_final_next_distances = torch.stack(
            [s for s, f in zip(batch.next_distances, batch.flags.unsqueeze(1)) if f == False]
        ).to(device)

        action_values = self.model(batch.walls, batch.preys, batch.agents, batch.distances
                                   ).gather(1, batch.actions.unsqueeze(1).type('torch.LongTensor').to(device)
                                            ).to(device)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(
                self.target_model(
                    non_final_next_walls, non_final_next_preys, non_final_next_agents, non_final_next_distances), axis=1)[0]

        expected_action_values = (next_state_values * GAMMA) + batch.rewards
        loss = self.loss(action_values, expected_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, walls, preys, agents, distances, target=False):
        walls = torch.as_tensor(walls).to(device).unsqueeze(1).unsqueeze(0)
        preys = torch.as_tensor(preys).to(device).unsqueeze(1).unsqueeze(0)
        agents = torch.as_tensor(agents).to(device).unsqueeze(1).unsqueeze(0)
        distances = torch.as_tensor(distances).to(device).unsqueeze(1).unsqueeze(0)

        if not target:
            actions = self.model(walls, preys, agents, distances)
        else:
            actions = self.target_model(walls, preys, agents, distances)

        return np.argmax(actions.cpu().detach().numpy())

    def update(self, transition):
        self.consume_transition(transition)

        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)

        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()

        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")
