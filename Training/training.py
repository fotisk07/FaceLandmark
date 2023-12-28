from tqdm import tqdm 
import torch
from utils import mask_to_landmarks
import wandb


class Trainer:
    def __init__(self, model, optimizer, criterion_mask, criterion_landmark, device, log_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion_mask = criterion_mask
        self.criterion_landmark = criterion_landmark
        self.device = device
        self.wandb = log_wandb

    def train(self, train_dataloader, valid_loader,  epochs, verbose=True):
        self.model.train()
        
        print("Starting training...")
        print("Training on: ", self.device)

        for e in range(epochs):
            running_loss = 0.0
            print(f"Epoch {e+1}/{epochs}")
            for i, sample in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                images_batch, landmarks_batch, masks_batch = sample['image'], sample['landmarks'], sample['mask']
                images_batch, masks_batch = images_batch.to(self.device), masks_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images_batch)
                loss = self.criterion_mask(masks_batch, outputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if self.wandb:
                    wandb.log({"Mask Loss": loss.item()})

            landmarks_pred = torch.tensor(mask_to_landmarks(outputs), dtype=torch.float32)
            landmarks_loss = self.criterion_landmark(landmarks_batch, landmarks_pred)

            if self.wandb:
                wandb.log({"Epochs" : e, "Landmark Loss: ": landmarks_loss.item()})

            if verbose:
                print(f"Epoch {e+1} Mask loss: {running_loss/len(train_dataloader)}")
                print(f"Epoch {e+1} Landmark loss: {landmarks_loss.item()}")

    
    