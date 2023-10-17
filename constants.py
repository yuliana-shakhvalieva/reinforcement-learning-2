# Set torch device for training
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed to fix random
SEED = 456

# Set rewards
CATCH_REWARD = 2000
STAND_PUNISHMENT = -1000
WALL_PUNISHMENT = -2000

# Set constants for DQN and reward counting
EPS = 0.1
GAMMA = 0.99

# Set training parameters
INITIAL_STEPS = 1024
TRANSITIONS = 500000

STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
BUFFER_MAXLEN = 2 ** 7

# Set information about environment
NUM_AGENTS = 1
NUM_ACTIONS = 5
NUM_PREYS = 100
FIELD_SIZE = 40 * 40
FIELS_SIDE = 40


