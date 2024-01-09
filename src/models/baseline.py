import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, num_landmarks):
        self.name = "Baseline"

        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding="same")
        self.conv2 = nn.Conv2d(10, num_landmarks, 1, padding="same")

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return torch.sigmoid(x)
    
