from itertools import count
from tqdm.notebook import trange
from world.utils import RenderedEnvWrapper
from tqdm.auto import trange
import numpy as np
import random
from state_utils import transform_state
from reward_utils import get_rewards
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from constants import SEED, NUM_AGENTS, INITIAL_STEPS

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class PositionTracker:
    def __init__(self):
        self.prev_position_1 = [-1 for _ in range(NUM_AGENTS)]
        self.prev_position_2 = [-1 for _ in range(NUM_AGENTS)]

        self.flag_position_1 = False
        self.flag_position_2 = False

    def reset(self, state, info):
        self.prev_position_1 = [-1 for _ in range(NUM_AGENTS)]
        self.prev_position_2 = [-1 for _ in range(NUM_AGENTS)]

        self.flag_position_1 = False
        self.flag_position_2 = False

        agents_info = np.array(info['predators'])

        for agent_info in agents_info:
            idx = agent_info['id']

            self.prev_position_2[idx] = (agent_info['x'], agent_info['y'])

        self.flag_position_2 = True

    def set(self, info):
        if self.flag_position_2 is True and self.flag_position_1 is False:
            agents_info = np.array(info['predators'])

            for agent_info in agents_info:
                idx = agent_info['id']

                self.prev_position_1[idx] = (agent_info['x'], agent_info['y'])

            self.flag_position_1 = True

        elif self.flag_position_2 is True and self.flag_position_1 is True:
            self.prev_position_2 = self.prev_position_1

            agents_info = np.array(info['predators'])

            for agent_info in agents_info:
                idx = agent_info['id']

                self.prev_position_1[idx] = (agent_info['x'], agent_info['y'])

            self.flag_position_1 = True

        else:
            raise Exception


def evaluate_policy(env, agent, tracker, episodes=5):
    env = RenderedEnvWrapper(env)

    returns = []
    scores = []
    difference = []
    for j in range(episodes):
        done = False
        state, info = env.reset()
        agent.reset(state, info)
        tracker.reset(state, info)

        agents_states = transform_state(state, info)
        total_reward = np.array([0. for _ in range(NUM_AGENTS)])

        while not done:
            actions = agent.select_action(agents_states, train=False).flatten()

            next_state, done, next_info = env.step(actions)
            tracker.set(next_info)
            next_agents_states = transform_state(next_state, next_info)
            rewards = get_rewards(info, next_info, actions, agent.distance_map, agent.action_map, tracker)

            total_reward += rewards
            agents_states = next_agents_states

        score = next_info['scores'][0]
        scores.append(score)
        difference.append(next_info['scores'][0] - next_info['scores'][1])
        returns.append(total_reward)
        env.render(score, f'./render/test/{j}')

    return np.array(returns), np.array(scores), np.array(difference)


def initialize_buffer(env, agent, target_agent, tracker):
    state, info = env.reset()
    target_agent.reset(state, 0)
    tracker.reset(state, info)

    agents_states = transform_state(state, info)

    for _ in trange(INITIAL_STEPS):
        target_action = target_agent.get_actions(state, 0)
        next_state, done, next_info = env.step(target_action)
        tracker.set(next_info)
        next_agents_states = transform_state(next_state, next_info)
        rewards = get_rewards(info, next_info, target_action, target_agent.distance_map, target_agent.action_map, tracker)

        agent.memory.push(agents_states, target_action, next_agents_states, rewards, done)

        if not done:
            agents_states = next_agents_states
        else:
            state, info = env.reset()
            target_agent.reset(state, 0)
            tracker.reset(state, info)

            agents_states = transform_state(state, info)


def one_epoch_train(env, agent, tracker):
    state, info = env.reset()
    agent.reset(state, info)
    tracker.reset(state, info)

    agents_states = transform_state(state, info)

    for _ in count():
        actions = agent.select_action(agents_states).flatten()
        next_state, done, next_info = env.step(actions)
        tracker.set(next_info)
        next_agents_states = transform_state(next_state, next_info)
        rewards = get_rewards(info, next_info, actions, agent.distance_map, agent.action_map, tracker)

        agent.memory.push(agents_states, actions, next_agents_states, rewards, done)

        if not done:
            agents_states = next_agents_states
        else:
            break

        agent.optimize_model()

    agent.fix_losses()


class Logger:
    def __init__(self):
        self.scores = []

    def remember_score(self, score):
        self.scores.append(score)

    def plot(self, agent):
        clear_output(True)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(agent.total_losses, label='Train')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(loc=0, fontsize=16)
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(self.scores)
        plt.xlabel('Evaluating episodes', fontsize=16)
        plt.ylabel('Mean score', fontsize=16)
        plt.grid()
        plt.show()


def logging(env, agent, logger, tracker, i_episode):
    rewards, scores, difference = evaluate_policy(env, agent, tracker, 5)
    mean_score = np.mean(scores)

    logger.remember_score(mean_score)
    logger.plot(agent)

    print()
    print(f"Epoch: {i_episode + 1}")
    print(f"Reward mean: {np.mean(rewards, axis=0).round(2)}")
    print(f"Reward std: {np.std(rewards, axis=0).round(2)}")
    print(f"Scores mean: {mean_score}")
    mean_diff = np.mean(difference)
    print(f"Mean difference me-bot: {mean_diff}")
    print()

    agent.save(round(mean_score, 2))
