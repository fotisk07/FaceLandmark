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

    def train(self, train_dataloader, valid_loader,  epochs, verbose=True, evaluate_every=1):
        self.model.train()
        
        print("Starting training...")
        print(f"Training on: {self.device}")

        mask_valid_loss, landmark_valid_loss = None, None

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

            if evaluate_every and e % evaluate_every == 0:
                mask_valid_loss, landmark_valid_loss = self.evaluate(valid_loader, verbose=verbose)

            if verbose:
                print(f"Training Mask Loss: {running_loss/len(train_dataloader):.3f}") 
                if evaluate_every and e % evaluate_every == 0:                      
                    print(f"Validation Mask loss: {mask_valid_loss:.3f} || Validation Landmark loss: {landmark_valid_loss:.1f}")
    

            if self.wandb and evaluate_every and e % evaluate_every == 0:
                wandb.log({"Epochs" : e+1, "Mask Valid Loss: ": mask_valid_loss, 
                           "Landmark Valid Loss: ": landmark_valid_loss})



        return {"Mask Train Loss": running_loss/len(train_dataloader),
                "Mask Valid Loss": mask_valid_loss, 
                "Landmark Valid Loss": landmark_valid_loss }



    def evaluate(self, valid_dataloader, verbose=True):
        self.model.eval()
        if verbose:
            print("Starting evaluation...")
        mask_loss = 0.0
        landmarks_loss = 0.0
        for sample in tqdm(valid_dataloader, disable=not verbose):
            images, landmarks, masks = sample['image'], sample['landmarks'], sample['mask']
            images, landmarks , masks = images.to(self.device), landmarks.to(self.device), masks.to(self.device)
            mask_pred = self.model(images)
            landmarks_pred = torch.tensor(mask_to_landmarks(mask_pred), dtype=torch.float32).to(self.device)
            running_landmarks_loss = self.criterion_landmark(landmarks, landmarks_pred)
            running_mask__loss = self.criterion_mask(masks, mask_pred)
            mask_loss += running_mask__loss.item()
            landmarks_loss += running_landmarks_loss.item()

        
        return mask_loss/len(valid_dataloader), landmarks_loss/len(valid_dataloader)







    
    