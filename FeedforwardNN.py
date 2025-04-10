import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),   # Input layer → Hidden layer 1
            nn.ReLU(),
            nn.Dropout(0.2),         # Dropout layer
            nn.Linear(128, 64),      # Hidden layer 1 → Hidden layer 2
            nn.ReLU(),
            nn.Dropout(0.2),         # Dropout layer
            nn.Linear(64, 10)        # Hidden layer 2 → Output layer (10 classes)
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        return self.model(x)
