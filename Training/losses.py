import torch

def landmark_loss(landmarks, landmarks_pred):
    return  torch.mean(torch.norm(landmarks-landmarks_pred, dim=2).sum(dim=1))

def mask_loss(mask, mask_pred):
    mask_loss = torch.nn.BCELoss()
    return mask_loss(mask_pred, mask) 
