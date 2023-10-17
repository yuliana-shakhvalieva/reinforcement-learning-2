# Set torch device for training
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed to fix random
SEED = 456

# Set rewards
CATCH_REWARD = 5.
STAND_PUNISHMENT = -1.
WALL_PUNISHMENT = -2.
GO_REWARD = 1.

# Set constants for DQN and reward counting
EPS = 0.2
GAMMA = 0.99

# Set training parameters
INITIAL_STEPS = 100000
TRANSITIONS = 3000000

STEPS_PER_UPDATE = 1024
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 4e-2
BUFFER_MAXLEN = 2 ** 7

# Set information about environment
NUM_AGENTS = 1
NUM_ACTIONS = 5
NUM_PREYS = 100
FIELS_SIDE = 40


