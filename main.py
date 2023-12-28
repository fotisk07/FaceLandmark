from dataset import FaceLandmarksDataset, ReduceLandmarks, ToTensor, Normalise
from Training.training import Trainer
from Training import losses
from models import Baseline
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from utils import show_predictions
import matplotlib.pyplot as plt
import wandb 
import argparse




parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--wandb', action='store_true', help='Log to wandb')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
parser.add_argument('--verbose', action='store_true', help='Show loss in terminal')

args = parser.parse_args()
epochs = args.epochs
log_wandb = args.wandb
verbose = args.verbose

if __name__ == "__main__":
    if log_wandb:
        run = wandb.init(project="Face-Landmarks")

    trans = transforms.Compose([ReduceLandmarks(20), Normalise(), ToTensor()])
    train_data = FaceLandmarksDataset(csv_file='face_landmarks_adj.csv',root_dir="helen1_new",train=True,transform=trans)
    valid_data = FaceLandmarksDataset(csv_file='face_landmarks_adj.csv',root_dir="helen1_new",train=False,transform=trans)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=4, shuffle=True)

    model = Baseline(10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    landmark_loss = losses.landmark_loss
    mask_loss = losses.mask_loss

    trainer = Trainer(model, optimizer, mask_loss, landmark_loss, "cpu", log_wandb=log_wandb)

    trainer.train(train_loader, valid_loader, epochs, verbose=verbose)

