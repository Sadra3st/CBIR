import torch
import torchvision.models as models
import numpy as np
from embedding.preprocess import load_image


def load_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model

def image_to_embedding(model, path):
    image = load_image(path)

    with torch.no_grad():
        embedding = model(image).squeeze(0).numpy()
    
    embedding = embedding.flatten()
    
    return embedding