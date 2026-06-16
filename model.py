import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=3):
    model = models.resnet18(pretrained=True)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model