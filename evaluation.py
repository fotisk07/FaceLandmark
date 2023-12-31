import torch
from tqdm import tqdm
from utils import mask_to_landmarks


class Evaluator:
    def __init__(self, model, criterion_mask, criterion_landmark, device):
        self.model = model
        self.criterion_mask = criterion_mask
        self.criterion_landmark = criterion_landmark
        self.device = device

    def evaluate(self, data, verbose=False):
        self.model.eval()
        if verbose:
            print("Starting evaluation...")

        mask_loss = 0.0
        landmarks_loss = 0.0
        for sample in tqdm(data, disable=not verbose):
            images, landmarks, masks = sample['image'], sample['landmarks'], sample['mask']
            images, landmarks , masks = images.to(self.device), landmarks.to(self.device), masks.to(self.device)
            mask_pred = self.model(images)
            landmarks_pred = mask_to_landmarks(mask_pred).float()
            running_landmarks_loss = self.criterion_landmark(landmarks, landmarks_pred)
            running_mask__loss = self.criterion_mask(masks, mask_pred)
            mask_loss += running_mask__loss.item()
            landmarks_loss += running_landmarks_loss.item()
        
        if verbose:
            print("Evaluation done!")
            print(f"Validation Mask loss: {mask_loss:.3f} || Validation Landmark loss: {landmarks_loss:.1f}")


        return mask_loss/len(data), landmarks_loss/len(data)
    