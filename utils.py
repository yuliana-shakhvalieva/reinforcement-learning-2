import numpy as np
import torch
import random
from world.envs import OnePlayerEnv
from world.realm import Realm
from world.map_loaders.base import MixedMapLoader
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from state_utils import get_relative_states
from reward_utils import get_rewards
from agent import Agent

from constants import SEED, NUM_AGENTS

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def evaluate_policy(main_agent, episodes=5):
    env = OnePlayerEnv(Realm(MixedMapLoader(
        (SingleTeamLabyrinthMapLoader(), SingleTeamRocksMapLoader())), 1, playable_team_size=NUM_AGENTS))
    returns = []
    scores = []
    for _ in range(episodes):
        done = False
        state, info = env.reset()
        walls, preys, agents, distances = get_relative_states(state, info)
        total_reward = np.array([0. for _ in range(NUM_AGENTS)])

        while not done:
            actions = []
            for i in range(NUM_AGENTS):
                actions.append(main_agent.act(walls[i], preys[i], agents[i], distances[i]))

            next_state, done, next_info = env.step(np.array(actions))
            next_walls, next_preys, next_agents, next_distances = get_relative_states(next_state, next_info)
            rewards = get_rewards(preys, next_preys, distances, next_distances, info, next_info, actions, walls)
            total_reward += rewards

            walls, preys, agents, distances = next_walls, next_preys, next_agents, next_distances

        scores.append(next_info['scores'][0])
        returns.append(total_reward)

    return np.array(returns), np.array(scores)


def test_reward():
    env = OnePlayerEnv(Realm(MixedMapLoader(
        (SingleTeamLabyrinthMapLoader(), SingleTeamRocksMapLoader())), 1, playable_team_size=NUM_AGENTS))

    done = False
    state, info = env.reset()
    walls, preys, other_agents = get_relative_states(state, info)
    main_agent = Agent()

    while not done:
        actions = []
        for i in range(NUM_AGENTS):
            actions.append(main_agent.act(walls[i], preys[i], other_agents[i]))

        next_state, done, next_info = env.step(np.array(actions))
        next_walls, next_preys, next_other_agents = get_relative_states(next_state, next_info)
        rewards = get_rewards(preys, next_preys, info, next_info)

        print(rewards)
        print(next_info['scores'][0])
        print()

        walls, preys, other_agents = next_walls, next_preys, next_other_agents
