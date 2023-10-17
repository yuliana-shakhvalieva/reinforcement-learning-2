import numpy as np

from constants import SEED, NUM_AGENTS, FIELS_SIDE

np.random.seed(SEED)


def divide(state, info):
    abs_walls, abs_preys = np.zeros(state.shape[:2]), np.zeros(state.shape[:2])
    dead_preys = np.array(info['eaten'].keys())

    abs_walls[np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == -1)] = 1
    abs_preys[np.logical_and(state[:, :, 0] == 1, state[:, :, :] not in dead_preys)] = 1

    return np.array(abs_walls), np.array(abs_preys)


def relate_coords(abs_coords, x, y):
    rel_coords = np.zeros_like(abs_coords)

    rel_coords[:FIELS_SIDE - x, :FIELS_SIDE - y] = abs_coords[x:, y:]
    rel_coords[:FIELS_SIDE - x, FIELS_SIDE - y:] = abs_coords[x:, :y]
    rel_coords[FIELS_SIDE - x:, :FIELS_SIDE - y] = abs_coords[:x, y:]
    rel_coords[FIELS_SIDE - x:, FIELS_SIDE - y:] = abs_coords[:x, :y]

    return np.array(rel_coords)


def get_relative_states(state, info):
    abs_walls, abs_preys = divide(state, info)
    rel_walls, rel_preys = [[]] * NUM_AGENTS, [[]] * NUM_AGENTS

    agents_coords = np.array(info['predators'])

    for agent_coord in agents_coords:
        x = agent_coord['x']
        y = agent_coord['y']
        idx = agent_coord['id']

        rel_walls[idx] = relate_coords(abs_walls, y, x)
        rel_preys[idx] = relate_coords(abs_preys, y, x)

    return np.array(rel_walls, dtype=np.float32), np.array(rel_preys, dtype=np.float32)
