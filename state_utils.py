import numpy as np
import copy

from constants import SEED, NUM_AGENTS, FIELS_SIDE

np.random.seed(SEED)


def divide(state, info):
    abs_walls, abs_preys, abs_agents = np.zeros(state.shape[:2]), np.zeros(state.shape[:2]), np.zeros(
        state.shape[:2])
    dead_preys = np.array(info['eaten'].keys())

    abs_walls[np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == -1)] = 1
    abs_preys[np.logical_and(state[:, :, 0] == 1, state[:, :, :] not in dead_preys)] = 1
    abs_agents[state[:, :, 0] == 0] = 1

    return np.array(abs_walls), np.array(abs_preys), np.array(abs_agents)


def relate_coords(abs_coords, x, y):
    rel_coords = np.zeros_like(abs_coords)

    rel_coords[:FIELS_SIDE - x, :FIELS_SIDE - y] = abs_coords[x:, y:]
    rel_coords[:FIELS_SIDE - x, FIELS_SIDE - y:] = abs_coords[x:, :y]
    rel_coords[FIELS_SIDE - x:, :FIELS_SIDE - y] = abs_coords[:x, y:]
    rel_coords[FIELS_SIDE - x:, FIELS_SIDE - y:] = abs_coords[:x, :y]

    return np.array(rel_coords)


def get_distance_map(state):
    mask = np.zeros(state.shape[:2], np.bool)
    mask[np.logical_or(np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == 0),
                       state[:, :, 0] >= 0)] = True
    mask = mask.reshape(-1)

    coords_amount = state.shape[0] * state.shape[1]
    distance_map = (coords_amount + 1) * np.ones((coords_amount, coords_amount))
    np.fill_diagonal(distance_map, 0.)
    distance_map[np.logical_not(mask)] = (coords_amount + 1)
    distance_map[:, np.logical_not(mask)] = (coords_amount + 1)

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
        old_distances = copy.deepcopy(distance_map)
        for j in range(coords_amount):
            if mask[j]:
                for i in indexes_helper[j]:
                    if mask[i]:
                        distance_map[j] = np.minimum(distance_map[j], distance_map[i] + 1)
        updated = (old_distances != distance_map).sum() > 0

    return distance_map


def get_relative_states(state, info):
    abs_walls, abs_preys, abs_agents = divide(state, info)
    distance_map = get_distance_map(state)
    rel_walls, rel_preys, rel_agents, rel_distances = [[]] * NUM_AGENTS, [[]] * NUM_AGENTS, [[]] * NUM_AGENTS, [
        []] * NUM_AGENTS

    agents_coords = np.array(info['predators'])

    for agent_coord in agents_coords:
        x = agent_coord['x']
        y = agent_coord['y']
        idx = agent_coord['id']
        flatten_idx = y * state.shape[1] + x

        rel_walls[idx] = relate_coords(abs_walls, y, x)
        rel_preys[idx] = relate_coords(abs_preys, y, x)
        rel_agents[idx] = relate_coords(abs_agents, y, x)
        abs_distances = distance_map[flatten_idx].reshape(state.shape[0], state.shape[1])
        rel_distances[idx] = relate_coords(abs_distances, y, x)

    return np.array(rel_walls, dtype=np.float32), np.array(rel_preys, dtype=np.float32), np.array(rel_agents,
                                                                                                  dtype=np.float32), np.array(
        rel_distances,
        dtype=np.float32)
