import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

def build_model(num_classes: int, pretrained: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, num_classes))
    return model

def freeze_backbone(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True
