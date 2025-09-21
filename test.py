import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms as T
import cv2
from PIL import Image

# ============================
# 1. Load pretrained model
# ============================
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

labels = weights.meta["categories"]

for label in labels: print(label)