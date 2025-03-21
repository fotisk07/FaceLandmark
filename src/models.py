import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_size):
        super(Baseline, self).__init__()
        self.fn1 = nn.Linear(input_size, 10)  # 2*5 for keypoints

        self.loss = nn.MSELoss()

    def forward(self, image, targets=None):
        image = image.flatten(1)
        output = self.fn1(image)  # (B, 14)

        keypoints = output.view(-1, 5, 2)  # Keypoints (B, 5, 2)

        loss = None
        if targets is not None:
            loss = self.loss(keypoints, targets[:, 1:, :2])

        return keypoints, loss
