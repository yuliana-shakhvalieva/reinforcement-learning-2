from tqdm import tqdm, trange
import numpy as np
import torch
import random
from agent import Agent
from utils import evaluate_policy
from world.envs import OnePlayerEnv
from world.realm import Realm
from world.map_loaders.base import MixedMapLoader
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from state_utils import get_relative_states
from reward_utils import get_rewards
from world.scripted_agents import ClosestTargetAgent

from constants import SEED, TRANSITIONS, INITIAL_STEPS, NUM_AGENTS, EPS, NUM_ACTIONS, device

print(device)

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

env = OnePlayerEnv(Realm(MixedMapLoader(
    (SingleTeamLabyrinthMapLoader(), SingleTeamRocksMapLoader())), 1, playable_team_size=NUM_AGENTS))
state, info = env.reset()

walls, preys = get_relative_states(state, info)

main_agent = Agent()

target_agent = ClosestTargetAgent()
target_agent.reset(state, 0)

for _ in tqdm(range(INITIAL_STEPS)):
    target_actions = target_agent.get_actions(state, 0)
    next_state, done, next_info = env.step(target_actions)
    next_walls, next_preys = get_relative_states(next_state, next_info)
    rewards = get_rewards(info, next_info, target_actions, walls)

    for (wall,
         prey,
         action,
         next_wall,
         next_prey,
         reward) \
            in zip(walls,
                   preys,
                   target_actions,
                   next_walls,
                   next_preys,
                   rewards):
        main_agent.update(
            (wall,
             prey,
             action,
             next_wall,
             next_prey,
             reward,
             done)
        )

    if not done:
        walls, preys = next_walls, next_preys
    else:
        state, info = env.reset()
        walls, preys = get_relative_states(state, info)

for j in trange(TRANSITIONS):
    # Epsilon-greedy policy
    actions = []
    for i in range(NUM_AGENTS):
        if random.random() < EPS:
            actions.append(random.randint(0, NUM_ACTIONS - 1))
        else:
            actions.append(main_agent.act(walls[i], preys[i]))

    actions = np.array(actions)
    next_state, done, next_info = env.step(actions)
    next_walls, next_preys = get_relative_states(next_state, next_info)
    rewards = get_rewards(info, next_info, actions, walls)

    for (wall,
         prey,
         action,
         next_wall,
         next_prey,
         reward) \
            in zip(walls,
                   preys,
                   actions,
                   next_walls,
                   next_preys,
                   rewards):
        main_agent.update(
            (wall,
             prey,
             action,
             next_wall,
             next_prey,
             reward,
             done)
        )

    if not done:
        walls, preys = next_walls, next_preys
    else:
        state, info = env.reset()
        walls, preys = get_relative_states(state, info)

    if (j + 1) % (TRANSITIONS // 10_000) == 0:
        rewards, scores = evaluate_policy(main_agent, 1)
        print(f"Step: {j + 1}")
        print(f"Reward mean: {np.mean(rewards, axis=0).round(2)}")
        print(f"Reward std: {np.std(rewards, axis=0).round(2)}")
        print(f"Scores mean: {np.mean(scores)}")
        print()
        main_agent.save()
