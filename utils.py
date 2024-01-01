import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_image_with_landmarks(image, landmarks, blob_s=10, text_size = 6, color ='b'):

    if torch.is_tensor(image):
        image = image.numpy().transpose((1, 2, 0))
    if torch.is_tensor(landmarks):
        landmarks = landmarks.numpy()

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = blob_s, marker = '.', c = color)
    for i in range(landmarks.shape[0]):
        plt.text(landmarks[i,0], landmarks[i,1], str(i+1), size=text_size)

def plot_image(image):
    if torch.is_tensor(image):
        if len(image.shape) == 2:
            image = image.detach().numpy()
        else:
            image = image.numpy().transpose((1, 2, 0))
    plt.imshow(image)

    
def mask_to_landmarks(a, device='cpu'):
    flattened = a.view(a.shape[0], a.shape[1], -1)

    maxis = torch.argmax(flattened, dim=-1)

    indices = torch.stack((maxis % torch.tensor([a.shape[2]]).to(device) , maxis // torch.tensor(a.shape[3]).to(device)), dim=-1)

    return indices
