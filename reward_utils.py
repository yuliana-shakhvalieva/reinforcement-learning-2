import numpy as np

from constants import SEED, NUM_AGENTS, CATCH_REWARD, WALL_PUNISHMENT, STAND_PUNISHMENT, GO_REWARD

np.random.seed(SEED)


def count_catch_rewards(info):
    rewards = [0 for _ in range(NUM_AGENTS)]

    good_agents = np.array(list(info['eaten'].values()))

    for team, idx in good_agents:
        rewards[idx] += 1

    return np.array(rewards) * CATCH_REWARD


def get_reward_catch(info, next_info):
    catch = count_catch_rewards(info)
    next_catch = count_catch_rewards(next_info)

    return next_catch - catch


def get_coords_after_action(action, shape_x, shape_y):
    if action == 0:
        return 0, 0
    elif action == 1:
        return 0, 1 % shape_y
    elif action == 2:
        return 0, (shape_x - 1) % shape_y
    elif action == 3:
        return (shape_x - 1) % shape_x, 0
    elif action == 4:
        return 1 % shape_x, 0


def get_punishment_go_on_wall(actions, walls):
    rewards = [0 for _ in range(NUM_AGENTS)]

    for idx, (action, wall) in enumerate(zip(actions, walls)):
        x, y = get_coords_after_action(action, wall.shape[0], wall.shape[1])

        if wall[x, y] == 1:
            rewards[idx] = 1

    return np.array(rewards) * WALL_PUNISHMENT


def get_punishment_stand(actions):
    return np.where(actions == 0, STAND_PUNISHMENT, 0)


def get_reward_go(actions):
    return np.where(actions != 0, GO_REWARD, 0)


def get_rewards(info, next_info, actions, walls):
    rewards = get_reward_catch(info, next_info)
    rewards += get_reward_go(actions)
    rewards += get_punishment_go_on_wall(actions, walls)
    rewards += get_punishment_stand(actions)

    return rewards
