# Model selection: 'vae' or 'autoregressive'
MODEL_TYPE = 'vae'

# Common parameters
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# VAE-specific parameters
BETA = 0.5
VAE_CHECKPOINT_DIR = 'dna_vae_checkpoints'

# Autoregressive-specific parameters
AUTOREGRESSIVE_CHECKPOINT_DIR = 'dna_autoregressive_checkpoints'
TEMPERATURE = 1.0

# Data paths
TRAIN_PATH = '/kaggle/input/dds-e-sim-small-clipped/train.csv'
TEST_PATH = '/kaggle/input/dds-e-sim-small-clipped/test.csv'
INFERENCE_OUTPUT_PATH = 'dna_inference_results.csv'

# Sequence parameters
MAX_SEQ_LEN = 147
VOCAB_SIZE = 7  # A, C, G, T, S, E, P

# Model architecture parameters
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
LATENT_DIM = 128  # For VAE only
