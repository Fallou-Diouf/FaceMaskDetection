import torch

from src.model import FastRCNN
from configs.config import NUM_CLASSES

from src.dataset import FaceMaskDataset
from configs.config import (
    DATA_FOLDER,
    BATCH_SIZE,
)

# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Model
model = FastRCNN(num_classes=NUM_CLASSES)

# Load trained weights
model.load_state_dict(
    torch.load(
        "outputs/models/model_epoch9.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

print(f"Device: {device}")
print("Model loaded successfully")