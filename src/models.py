import torch
import numpy as np
import torch.nn as nn

tensor = torch.Tensor
array = np.ndarray


class Baseline(nn.Module):
    def __init__(self, input_size):
        super(Baseline, self).__init__()
        self.fn1 = nn.Linear(input_size, 10)  # 2*5 for keypoints

        self.loss = nn.MSELoss()

    def forward(self, image, targets=None) -> tuple[tensor, tensor]:
        image = image.flatten(1)
        output = self.fn1(image)  # (B, 14)

        keypoints = output.view(-1, 5, 2)  # Keypoints (B, 5, 2)

        loss = None
        if targets is not None:
            loss = self.loss(keypoints, targets[:, 1:, :2])

        return keypoints, loss

    def generate(self, input: tensor | dict) -> array:
        image = input
        if isinstance(input, dict):
            if not "image" in input:
                raise KeyError
            image = input["image"]

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        keypoints, _ = self(image)

        return keypoints.detach().cpu().numpy()
