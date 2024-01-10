import copy
import math
import os

import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch

from net import PolicyNet
from buffer import ReplayMemory, Transition

from constants import SEED, LEARNING_RATE, BUFFER_MAXLEN, EPS_END, EPS_START, EPS_DECAY, NUM_ACTIONS, NUM_AGENTS, DEVICE, BATCH_SIZE, GAMMA, GRAD_CLIP, TAU, TRANSITIONS, INITIAL_STEPS

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class DQN:
    def __init__(self):
        self.learning_steps_done = 0

        self.policy_net = PolicyNet()
        self.target_net = PolicyNet()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        # self.optimizer = optim.Adagrad(self.policy_net.parameters(), lr=LEARNING_RATE)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, (TRANSITIONS+INITIAL_STEPS), eta_min=1e-4)

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()

        self.memory = ReplayMemory(BUFFER_MAXLEN)
        self.batch_losses = []
        self.total_losses = []

        self.distance_map = None
        self.action_map = None

    def select_action(self, agents_states, train=True):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learning_steps_done / EPS_DECAY)

        if train:
            self.learning_steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                actions = self.policy_net.get_q_values(torch.from_numpy(np.array([agents_states])).to(DEVICE)).max(2)[1]
                return actions.detach().cpu().numpy()
        else:
            return np.random.randint(NUM_ACTIONS, size=NUM_AGENTS)

    def __sample_batch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        return batch

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.__sample_batch()

        non_final_mask = torch.tensor(tuple(map(lambda s: s == False, batch.flag)),
                                      device=DEVICE,
                                      dtype=torch.bool)

        non_final_next_states = torch.stack(
            [s for s, f in zip(torch.tensor(np.array(batch.next_agents_states)),
                               torch.tensor(np.array(batch.flag))) if f == False], dim=0).to(DEVICE)

        agents_states_batch = torch.tensor(np.array(batch.agents_states)).to(DEVICE)
        action_batch = torch.tensor(np.array(batch.action)).to(DEVICE)
        reward_batch = torch.tensor(np.array(batch.reward)).to(DEVICE)

        q_values = self.policy_net.get_q_values(agents_states_batch)
        state_action_values = q_values.gather(2, action_batch.unsqueeze(2).long())

        next_state_values = torch.zeros((BATCH_SIZE, NUM_AGENTS), device=DEVICE)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net.get_q_values(non_final_next_states).max(2)[0]

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.criterion(state_action_values.float(), expected_state_action_values.unsqueeze(2).float())

        self.batch_losses.append(loss.data.item())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()
        self.scheduler.step()

        self.__target_net_update()

    def __target_net_update(self):
        with torch.no_grad():
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            self.target_net.load_state_dict(target_net_state_dict)

    def save(self, reward):
        os.makedirs('./agents', exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"./agents/agent_{reward}.pkl")

    def fix_losses(self):
        mean_batch_losses = sum(self.batch_losses) / len(self.batch_losses)
        self.total_losses.append(mean_batch_losses)
        self.batch_losses = []

    def reset(self, state, info):
        mask = np.zeros(state.shape[:2], np.bool)
        mask[np.logical_or(np.logical_and(state[:, :, 0] == -1, state[:, :, 1] >= 0),
                           state[:, :, 0] >= 0)] = True
        mask = mask.reshape(-1)

        coords_amount = state.shape[0] * state.shape[1]
        self.distance_map = (coords_amount + 1) * np.ones((coords_amount, coords_amount))
        np.fill_diagonal(self.distance_map, 0.)
        self.distance_map[np.logical_not(mask)] = (coords_amount + 1)
        self.distance_map[:, np.logical_not(mask)] = (coords_amount + 1)

        indexes_helper = [
            [
                x * state.shape[1] + (y + 1) % state.shape[1],
                x * state.shape[1] + (state.shape[1] + y - 1) % state.shape[1],
                ((state.shape[0] + x - 1) % state.shape[0]) * state.shape[1] + y,
                ((x + 1) % state.shape[0]) * state.shape[1] + y
            ]
            for x in range(state.shape[0]) for y in range(state.shape[1])
        ]

        updated = True
        while updated:
            old_distances = copy.deepcopy(self.distance_map)
            for j in range(coords_amount):
                if mask[j]:
                    for i in indexes_helper[j]:
                        if mask[i]:
                            self.distance_map[j] = np.minimum(self.distance_map[j], self.distance_map[i] + 1)
            updated = (old_distances != self.distance_map).sum() > 0

        self.action_map = np.zeros((coords_amount, coords_amount), int)
        for j in range(coords_amount):
            self.action_map[j] = np.argmin(np.stack([self.distance_map[i] + 1 for i in indexes_helper[j]], axis=1),
                                           axis=1) + 1
