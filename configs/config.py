from pathlib import Path


# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent


# Dataset
DATA_FOLDER = ROOT_DIR / "data"


# Training
BATCH_SIZE = 2
NUMBER_EPOCHS = 5

TRAIN_SPLIT_PERCENTAGE = 0.6
VAL_SPLIT_PERCENTAGE = 0.2
TEST_SPLIT_PERCENTAGE = 0.2


# Model
NUM_CLASSES = 4

# Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.01

SEED = 42
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5