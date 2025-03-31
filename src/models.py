import torch
import numpy as np
import torch.nn as nn

from abc import ABC, abstractmethod

tensor = torch.Tensor
array = np.ndarray


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, images, targets=None):
        pass

    def generate(self, input: tensor | dict) -> array:
        image = input
        if isinstance(input, dict):
            if "image" not in input:
                raise KeyError("Missing 'image' key in input dictionary.")
            image = input["image"]

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        keypoints, _ = self(image)

        return keypoints.detach().cpu().numpy()


class Baseline(Model):
    def __init__(self, input_size):
        super().__init__()
        self.fn1 = nn.Linear(input_size, 10)  # 2*5 for keypoints

        self.loss = nn.MSELoss()

    def forward(self, images, targets=None) -> tuple[tensor, tensor]:
        images = images.flatten(1)
        output = self.fn1(images)  # (B, 14)

        keypoints = output.view(-1, 5, 2)  # Keypoints (B, 5, 2)

        loss = None
        if targets is not None:
            loss = self.loss(keypoints, targets[:, 1:, :2])

        return keypoints, loss


class ScaledBaseline(Model):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        self.loss = nn.MSELoss()

    def forward(self, images, targets=None):
        images = images.flatten(1)
        output = self.layers(images)  # (B, 14)

        keypoints = output.view(-1, 5, 2)  # Keypoints (B, 5, 2)

        loss = None
        if targets is not None:
            loss = self.loss(keypoints, targets[:, 1:, :2])

        return keypoints, loss


class Convolution(Model):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 128, 128)
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 64, 64)
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 32, 32)
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 16, 16)
            # Conv Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 8, 8)
            nn.Flatten(),  # Output: (512 * 8 * 8 = 32768)
            # Fully Connected Layers
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(1024, 10),  # 5 keypoints (x, y) → 10 values
        )
        self.loss = nn.MSELoss()

    def forward(self, images, targets=None):
        keypoints = self.layers(images).view(-1, 5, 2)  # Reshape to (B, 5, 2)

        loss = None
        if targets is not None:
            loss = self.loss(keypoints, targets[:, 1:, :2])

        return keypoints, loss
