from tqdm import tqdm 
import torch
from src.utils import mask_to_landmarks
import wandb
import os

class Trainer:
    def __init__(self, model, optimizer, criterion_mask, criterion_landmark, device, log_wandb=False, save=False,
                 save_every=None, evaluator=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion_mask = criterion_mask
        self.criterion_landmark = criterion_landmark
        self.device = device
        self.wandb = log_wandb
        self.save = save
        self.save_every = save_every
        self.evaluator = evaluator


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

            if evaluate_every and e % evaluate_every == 0 and self.evaluator:
                if verbose:
                    print("Starting evaluation...")
                mask_valid_loss, landmarks_valid_loss = self.evaluator.evaluate(valid_loader, verbose=verbose)
                if verbose:
                    print("Evaluation finished")
                    print(f"Validation Mask loss: {mask_valid_loss:.3f} || Validation Landmark loss: {landmarks_valid_loss:.1f}")
                    
            if verbose:
                print(f"Training Mask Loss: {running_loss/len(train_dataloader):.3f}") 
                

            if self.wandb:
                wandb.log({"Epochs" : e+1, "Mask Valid Loss: ": mask_valid_loss, 
                            "Landmark Valid Loss: ": landmarks_valid_loss})

            if self.save and self.save_every and e % self.save_every == 0:
                self.save_model(e+1)


        if self.save:
            self.save_model(epochs, extra_name="final")


        return {"Mask Train Loss": running_loss/len(train_dataloader),
                "Mask Valid Loss": mask_valid_loss, 
                "Landmark Valid Loss": landmark_valid_loss }



    def save_model(self, epochs, extra_name=""):

        dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs": epochs,
        }
        
        path = f"model_weights/{self.model.name}"

        # check if directory exists
        if not os.path.exists(path):
            os.makedirs(path)

            
        if extra_name:
            path += f"/{epochs}_{extra_name}.pth"
        else:   
            path += f"/{epochs}.pth"



        torch.save(dict, path)
        print("Model saved successfully")

        






    
    