import torch

# Set device cuda for GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train Hyperparameters
TRAIN_DIR = "/home/rodrigoduarte/Documentos/projeto/imagens/swedish/train"
TEST_DIR = "/home/rodrigoduarte/Documentos/projeto/imagens/swedish/test"
SHAPE = (384, 384)
MAX_EPOCHS = 10
NUM_CLASSES = 15
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-8
OPTIMIZER_MOMENTUM = (0.9, 0.999)
SCALE_FACTOR = 0.8 
DROP_PATH_RATE = 0.1
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 6

ACCELERATOR = 'gpu'
DEVICES = 2
PRECISION = '16-mixed'