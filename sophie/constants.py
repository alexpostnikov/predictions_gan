OBS_LEN = 11
PRED_LEN = 16
NUM_WORKERS = 0

MLP_DIM = 64
H_DIM = 256
EMBEDDING_DIM = 32
BOTTLENECK_DIM = 64
NOISE_DIM = 32

DATASET_NAME = 'eth'
NUM_ITERATIONS = 20000
NUM_EPOCHS = 500
G_LR = 1e-4
D_LR = 1e-3
G_LR_DECAY_STEP = 3000
G_LR_DECAY_RATE = 0.5
G_STEPS = 1
D_STEPS = 1

MAX_PEDS = 128
BEST_K = 4
PRINT_EVERY = 1000  # batch
NUM_MODES = 5
NUM_SAMPLES_CHECK = 2000  # samples

ATTN_L = 90
ATTN_D = 512  # cnn out shape
ATTN_D_DOWN = 32

NLL_LOSS_COEF = 0
DIVERSITY_LOSS_COEF = 0