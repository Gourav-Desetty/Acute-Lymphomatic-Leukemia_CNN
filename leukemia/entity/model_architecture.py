from pathlib import Path
import torch 
from torch import nn
from torchvision import models
from leukemia.logging.logger import logging 
from leukemia.constant.training_pipeline import DEVICE
from leukemia.constant.training_pipeline import MODEL_PATH

class LeukemiaCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True) :
        super().__init__()

        # self.model = models.densenet121(pretrained=pretrained)
        self.model = models.efficientnet_b0(pretrained=pretrained)

        # num_features = self.model.classifier.in_features
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.4),  # Reduced from 0.6
        #     nn.Linear(num_features, 256),  # Increased capacity
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),  # Added batch norm
        #     nn.Dropout(0.3),  # Reduced from 0.4
        #     nn.Linear(256, num_classes)
        # )

        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),  #0.6
            nn.Linear(in_features=num_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),  #0.4
            nn.Linear(128, num_classes)
        )

    def forward(self, X):
        return self.model(X)

def save_model(model, model_name="leukemia_model_efficientnet_b0_01.pth", model_dir="../Models"):
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True, parents=False)
    model_save_path = model_path / model_name
    torch.save(model.state_dict(), model_save_path)
    return model_save_path

def load_model(path: str = MODEL_PATH) -> nn.Module:
    model = LeukemiaCNN(num_classes=2, pretrained=False)
    if path:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    logging.info("Loaded weights  from {path}")
    model.to(DEVICE)
    model.eval()

    return model