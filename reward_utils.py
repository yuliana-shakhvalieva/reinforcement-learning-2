import numpy as np

from constants import (SEED, NUM_AGENTS, CATCH_PREY_REWARD, CATCH_ENEMY_REWARD, GO_REWARD, FIELD_SIDE, CYCLE_PUNISHMENT,
                       STAND_PUNISHMENT, NUM_TEAMS, WALL_PUNISHMENT, KILL_PUNISHMENT, GET_BONUS_REWARD, LOSE_BONUS_PUNISHMENT, GO_TO_NEAREST_REWARD)

np.random.seed(SEED)


def get_reward_catch(info):
    rewards_preys = [0 for _ in range(NUM_AGENTS)]
    rewards_enemy = [0 for _ in range(NUM_AGENTS)]
    good_agents = [0 for _ in range(NUM_AGENTS)]

    eaten = info['eaten']

    for key, value in eaten.items():
        dead_team, dead_idx = key
        good_team, good_idx = value

        if good_team == 0:

            if dead_team == NUM_TEAMS:
                rewards_preys[good_idx] += 1

            elif 0 < dead_team < NUM_TEAMS:
                rewards_enemy[good_idx] += 1

            good_agents[good_idx] = 1

    rewards_preys = np.array(rewards_preys) * CATCH_PREY_REWARD
    rewards_enemy = np.array(rewards_enemy) * (CATCH_ENEMY_REWARD + max(0, 0.1 * (info['scores'][1] - info['scores'][0])))
    total_reward = rewards_preys + rewards_enemy

    return total_reward, good_agents


def get_reward_go(info, next_info, actions):
    rewards = [0. for _ in range(NUM_AGENTS)]

    agents_info = np.array(info['predators'])
    next_agents_info = np.array(next_info['predators'])

    for action, agent_info, next_agent_info in zip(actions, agents_info, next_agents_info):
        idx = agent_info['id']

        if action == 0:
            rewards[idx] = STAND_PUNISHMENT

        elif agent_info['x'] == next_agent_info['x'] and agent_info['y'] == next_agent_info['y']:
            rewards[idx] = WALL_PUNISHMENT

        else:
            rewards[idx] = GO_REWARD

    return np.array(rewards)


def get_punishment_kill(info):
    rewards = [0 for _ in range(NUM_AGENTS)]

    agents = np.array(info['predators'])

    for agent in agents:
        alive = agent['alive']
        idx = agent['id']

        if not alive:
            rewards[idx] = KILL_PUNISHMENT

    return np.array(rewards)


def get_reward_bonus(info, next_info):
    rewards = [0 for _ in range(NUM_AGENTS)]
    good_agents = [0 for _ in range(NUM_AGENTS)]

    agents = np.array(info['predators'])
    next_agents = np.array(next_info['predators'])

    for agent, next_agent in zip(agents, next_agents):
        idx = agent['id']

        bonus = agent['bonus_count']
        next_bonus = next_agent['bonus_count']

        if next_bonus > bonus:
            rewards[idx] += GET_BONUS_REWARD
            good_agents[idx] = 1

        elif next_bonus < bonus:
            rewards[idx] += LOSE_BONUS_PUNISHMENT

    return np.array(rewards), good_agents


def get_reward_go_to_nearest(info, actions, distance_map, action_map):
    rewards = [0 for _ in range(NUM_AGENTS)]
    good_agents = [0 for _ in range(NUM_AGENTS)]

    agents = np.array(info['predators'])
    preys = np.array(info['preys'])

    if len(preys) > 0:
        for agent in agents:
            ys = agent['x']
            xs = agent['y']
            idx = agent['id']

            target_x = preys[0]['y']
            target_y = preys[0]['x']

            for prey in preys:
                px = prey['y']
                py = prey['x']

                if (distance_map[xs * FIELD_SIDE + ys, px * FIELD_SIDE + py] <
                        distance_map[xs * FIELD_SIDE + ys, target_x * FIELD_SIDE + target_y]):
                    target_x = px
                    target_y = py

            if actions[idx] == action_map[xs * FIELD_SIDE + ys, target_x * FIELD_SIDE + target_y]:
                rewards[idx] += GO_TO_NEAREST_REWARD
                good_agents[idx] = 1

    return np.array(rewards), good_agents


def get_punishment_cycle(tracker, info, good_agents):
    rewards = [0. for _ in range(NUM_AGENTS)]
    prev_position_2 = tracker.prev_position_2

    agents_info = np.array(info['predators'])

    for agent_info in agents_info:
        idx = agent_info['id']

        if prev_position_2[idx] == (agent_info['x'], agent_info['y']) and good_agents[idx] == 0:
            rewards[idx] += CYCLE_PUNISHMENT

    return np.array(rewards)


def get_rewards(info, next_info, actions, distance_map, action_map, tracker):
    rewards, good_agents = get_reward_catch(next_info)

    rewards += get_reward_go(info, next_info, actions)

    rewards += get_punishment_kill(next_info)

    rew, ga = get_reward_bonus(info, next_info)
    rewards += rew
    good_agents += ga

    rew, ga = get_reward_go_to_nearest(info, actions, distance_map, action_map)
    rewards += rew
    good_agents += ga

    rewards += get_punishment_cycle(tracker, next_info, good_agents)

    return np.array(rewards)
