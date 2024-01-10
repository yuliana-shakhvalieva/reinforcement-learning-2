import torch

# Set torch device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed to fix random
SEED = 456

# Non-changeable parameters
NUM_ACTIONS = 5
FIELD_SIDE = 40

# Set information about environment
NUM_AGENTS = 1
NUM_TEAMS = 2
VIEW_FIELD = FIELD_SIDE // 4

# Set rewards
CATCH_PREY_REWARD = 1.
CATCH_ENEMY_REWARD = 3.
GET_BONUS_REWARD = 0.5

STAND_PUNISHMENT = -0.5
WALL_PUNISHMENT = -0.5
CYCLE_PUNISHMENT = -0.5

KILL_PUNISHMENT = -0.5
LOSE_BONUS_PUNISHMENT = -0.5

GO_REWARD = 0.0
GO_TO_NEAREST_REWARD = 0.01

# Set training parameters
INITIAL_STEPS = 10_000
TRANSITIONS = 50_000

BATCH_SIZE = 256
BUFFER_MAXLEN = 500_000

GAMMA = 0.99
TAU = 0.005

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10_000

LEARNING_RATE = 4e-5
GRAD_CLIP = 100
