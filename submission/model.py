import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5

def get_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    return model

def get_transform():
    return models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
