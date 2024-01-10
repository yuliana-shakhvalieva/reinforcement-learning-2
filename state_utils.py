import numpy as np

from constants import SEED, NUM_AGENTS, FIELD_SIDE, NUM_TEAMS, VIEW_FIELD

np.random.seed(SEED)


def divide(state):
    abs_walls = np.zeros(state.shape[:2])
    abs_preys = np.zeros(state.shape[:2])
    abs_enemies = np.zeros(state.shape[:2])
    abs_bonuses = np.zeros(state.shape[:2])
    abs_agents = np.zeros(state.shape[:2])

    abs_walls[np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == -1)] = 1
    abs_preys[state[:, :, 0] == NUM_TEAMS] = 1
    abs_enemies[np.logical_and(state[:, :, 0] > 0, state[:, :, 0] < NUM_TEAMS)] = 1
    abs_bonuses[np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == 1)] = 1
    abs_agents[state[:, :, 0] == 0] = 1

    return (np.array(abs_walls),
            np.array(abs_preys),
            np.array(abs_enemies),
            np.array(abs_bonuses),
            np.array(abs_agents))


def relate_coords(abs_coords, x, y):
    return np.roll(abs_coords, shift=(FIELD_SIDE // 2 - x, FIELD_SIDE // 2 - y), axis=(0, 1))


def cut_view_field(rel_coord):
    start_index = FIELD_SIDE // 2 - VIEW_FIELD // 2
    return rel_coord[start_index: start_index + VIEW_FIELD, start_index: start_index + VIEW_FIELD]


def relate_and_cut(abs_coords, x, y):
    rel_coord = relate_coords(abs_coords, x, y)
    return cut_view_field(rel_coord)


def get_agent_bonus(bonus):
    x = VIEW_FIELD // 2
    y = VIEW_FIELD // 2

    my_bonus = np.zeros((VIEW_FIELD, VIEW_FIELD))
    my_bonus[x, y] = bonus

    return my_bonus


def get_scores(score):
    x = VIEW_FIELD // 2
    y = VIEW_FIELD // 2

    scores = np.zeros((VIEW_FIELD, VIEW_FIELD))
    scores[x, y] = score - 1

    return scores


def get_relative_states(abs_walls, abs_preys, abs_enemies, abs_bonuses, abs_agents, agents_info, score_diff):
    rel_walls = [[]] * NUM_AGENTS
    rel_preys = [[]] * NUM_AGENTS
    rel_enemies = [[]] * NUM_AGENTS
    rel_bonuses = [[]] * NUM_AGENTS
    rel_agents = [[]] * NUM_AGENTS

    for agent_info in agents_info:
        x = agent_info['x']
        y = agent_info['y']
        idx = agent_info['id']
        # bonus = agent_info['bonus_count']

        rel_walls[idx] = relate_and_cut(abs_walls, y, x)
        rel_preys[idx] = relate_and_cut(abs_preys, y, x)
        rel_enemies[idx] = relate_and_cut(abs_enemies, y, x)
        rel_bonuses[idx] = relate_and_cut(abs_bonuses, y, x)
        rel_agents[idx] = relate_and_cut(abs_agents, y, x)

        rel_agents[idx] += get_scores(score_diff)

        # rel_agents[idx] += get_agent_bonus(bonus)

    return (np.array(rel_walls, dtype=np.float32),
            np.array(rel_preys, dtype=np.float32),
            np.array(rel_enemies, dtype=np.float32),
            np.array(rel_bonuses, dtype=np.float32),
            np.array(rel_agents, dtype=np.float32))


def transform_state(state, info):
    agents_info = np.array(info['predators'])
    score_diff = max(0, 0.1 * (info['scores'][1] - info['scores'][0]))

    abs_walls, abs_preys, abs_enemies, abs_bonuses, abs_agents = divide(state)
    rel_walls, rel_preys, rel_enemies, rel_bonuses, rel_agents = get_relative_states(abs_walls, abs_preys, abs_enemies, abs_bonuses, abs_agents, agents_info, score_diff)

    agents_states = [[]] * NUM_AGENTS

    for agent_info in agents_info:
        idx = agent_info['id']

        rel_wall = rel_walls[idx]
        rel_prey = rel_preys[idx]
        rel_enemy = rel_enemies[idx]
        rel_bonus = rel_bonuses[idx]
        rel_agent = rel_agents[idx]

        agents_states[idx] = np.stack((rel_wall, rel_prey, rel_enemy, rel_bonus, rel_agent), axis=0)

    return np.array(agents_states)
