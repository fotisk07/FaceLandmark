from dataset import FaceLandmarksDataset, ReduceLandmarks, ToTensor, Normalise
from Training.training import Trainer
from Training import losses
from models import Baseline, Autoencoder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from utils import show_predictions
import matplotlib.pyplot as plt
import wandb 
import argparse
import yaml



parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--wandb', action='store_true', help='Log to wandb')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
parser.add_argument('--verbose', action='store_true', help='Show loss in terminal')
parser.add_argument('--cuda', action='store_true', help='Use cuda')

args = parser.parse_args()
epochs = args.epochs
log_wandb = args.wandb
verbose = args.verbose
cuda = args.cuda    

# Load config file
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


lr = config["lr"]
batch_size = config["batch_size"]
csv_file_path = config["csv_file_path"]
root_dir = config["root_dir"]
evaluate_every = config["evaluate_every"]
save_model = config["save_model"]
save_every = config["save_every"]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if cuda else 'cpu'


if __name__ == "__main__":

    if log_wandb:
        run = wandb.init(project="Face-Landmarks")

    trans = transforms.Compose([ReduceLandmarks(20), Normalise(), ToTensor()])
    train_data = FaceLandmarksDataset(csv_file=csv_file_path,root_dir=root_dir,train=True,transform=trans)
    valid_data = FaceLandmarksDataset(csv_file=csv_file_path,root_dir=root_dir,train=False,transform=trans)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    model = Autoencoder(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    landmark_loss = losses.landmark_loss
    mask_loss = losses.mask_loss

    trainer = Trainer(model, optimizer, mask_loss, landmark_loss, device=device, log_wandb=log_wandb,
                      save=save_model, save_every=save_every)

    trainer.train(train_loader, valid_loader, epochs, verbose=verbose, evaluate_every=evaluate_every)

