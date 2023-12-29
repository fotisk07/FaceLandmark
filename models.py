import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, num_landmarks):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding="same")
        self.conv2 = nn.Conv2d(10, num_landmarks, 1, padding="same")

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return torch.sigmoid(x)
    

    

class Autoencoder(nn.Module):
    def __init__(self, num_landmarks):
        super(Autoencoder, self).__init__()

        # Image (3, 256,256)
        ### Encoder
        # Conv 1 (16, 256, 256) kernel 7, padding same
        # MaxPool 1 (10, 128, 128) 
        # Conv 2 (32, 128, 128) kernel 5, padding same
        # MaxPool 2 (32, 64, 64)
        # Conv 3 (64, 64, 64) kernel 3, padding same
        # MaxPool 3 (64, 32, 32)
        # Conv 4 (128, 32, 32) kernel 1, padding same
        # Max pool 4 (128, 16, 16) 

        ### Decoder
        # conv 5 (64, 32, 32)
        # conv 6 (32, 64, 64)
        # conv 7 (16, 128, 128)
        # conv 8 (1, 256, 256)
        self.name = "Autoencoder"

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(64,32, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(32,16, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(16,num_landmarks, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        
        

