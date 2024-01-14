import torch
import torch.nn as nn
import os

class Baseline(nn.Module):
    def __init__(self, num_landmarks, name="Default", gen=1):
        self.name = name
        self.gen = gen

        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding="same")
        self.conv2 = nn.Conv2d(10, num_landmarks, 1, padding="same")

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return torch.sigmoid(x)
    
    def save_model(self, epochs):
        dict = self.state_dict()

        path = f"model_weights/{self.name}/{self.gen}"

        # check if directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        path = f"{path}/{epochs}.pth"


        torch.save(dict, path)
        print("Model saved successfully")

