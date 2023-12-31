import torch
import matplotlib.pyplot as plt
import numpy as np

def show_image_landmarks(image, landmarks):

    if torch.is_tensor(image):
        image = image.numpy().transpose((1, 2, 0))
        landmarks = landmarks.numpy()

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')

def show_image_landmarks_batch(images, landmarks, x, y):
    for i in range(len(images)):
        plt.subplot(x,y, i+1)
        show_image_landmarks(images[i], landmarks[i])


def show_masks_batch(masks, x, y):
    # mask is (batch, 64, 256, 256)
    for i in range(len(masks)):
        plt.subplot(x,y, i+1)
        plt.imshow(masks[i].sum(dim=0).numpy())

def show_everything_batch(images, landmarks, masks):
    points = 7
    len_batch = len(images)

    for i in range(len_batch):
        plt.subplot(len_batch, points, points*i + 1 )
        plt.imshow(images[i].numpy().transpose(1,2,0))
        plt.scatter(landmarks[i,:,0], landmarks[i,:,1], s=10, marker='.', c='r')

    for i in range(len_batch):
        for j in range(points-1):
            plt.subplot(len_batch, points, points*i + j + 2)
            plt.imshow(masks[i,j].numpy())



def show_predictions(model, dataloader, device):
    model.eval()
    for sample in dataloader:
        images_batch, landmarks_batch, masks_batch = sample['image'], sample['landmarks'], sample['mask']
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        outputs = model(images_batch)

        new_landmarks = mask_to_landmarks(outputs)
        
        new_sample = {'image': images_batch, 'landmarks': new_landmarks, 'mask': outputs}
        show_everything_batch(new_sample)
        break


    
def mask_to_landmarks(a):
    flattened = a.view(a.shape[0], a.shape[1], -1)

    maxis = torch.argmax(flattened, dim=-1)

    indices = torch.stack((maxis % torch.tensor([a.shape[2]]) , maxis // torch.tensor(a.shape[3])), dim=-1)

    return indices