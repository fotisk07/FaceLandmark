import torch
from tqdm import tqdm
from utils import mask_to_landmarks, plot_image_with_landmarks, plot_image
import matplotlib.pyplot as plt
import numpy as np


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
    
    def create_img_landmarks_graph(self, data, num_images=1, show=True, save=False, save_path=None):
        """
        data : dataloader type of certain batch size. 
        If batch_size > num_images, then the first num_images images will be used.
        If batch_size < num_images, then the first batch_size images will be used.
        """
        self.model.eval()
        
        if num_images > len(data):
            num_images = len(data)

        sample = next(iter(data))
        images, landmarks, masks = sample['image'][:num_images], sample['landmarks'][:num_images], sample['mask'][:num_images]
        images, landmarks , masks = images.to(self.device), landmarks.to(self.device), masks.to(self.device)
        mask_pred = self.model(images)
        landmarks_pred = mask_to_landmarks(mask_pred).float()

        plt.figure(figsize=(10, 10))
        
        for i in range(0, 2 * num_images,2):
            plt.subplot(num_images, 2, i+1)
            plot_image_with_landmarks(images[i//2], landmarks[i//2], color='g')
            plt.axis('off')

            plt.subplot(num_images, 2,  i+2)
            plot_image_with_landmarks(images[i//2], landmarks_pred[i//2], color='r')
            plt.axis('off')

        if show:
            plt.show()
        
        if save:
            path = f'{save_path}/images_with_landmarks.png'
            plt.savefig(save_path)

    def create_landmark_comparaison_graph(self, data, num_landmarks=5 , num_images = 1, show=True, save=False, save_path=None):
        self.model.eval()
        sample = next(iter(data))
        images, landmarks, masks = sample['image'][:num_images], sample['landmarks'][:num_images], sample['mask'][:num_images]
        images, landmarks , masks = images.to(self.device), landmarks.to(self.device), masks.to(self.device)
        mask_pred = self.model(images)
        landmarks_pred = mask_to_landmarks(mask_pred).float()

        plt.figure(figsize=(10, 10))

        for j in range(num_images):
            for i in range(num_landmarks):
                plt.subplot(num_images,num_landmarks, j*num_landmarks + i+1)
                plot_image(images[j])
                plt.plot(landmarks[j, i, 0], landmarks[j, i,1], marker='o', color='green')
                plt.plot(landmarks_pred[j, i,0], landmarks_pred[j, i,1], marker='o', color='red')
                plt.title("Landmark {}".format(i+1))

                norm = np.linalg.norm(landmarks[j, i]-landmarks_pred[j, i])  
                plt.xlabel(f'Difference {norm:.2f}', size=10)
                plt.xticks([])
                plt.yticks([])

        if show:
            plt.show()

        if save:
            path = f'{save_path}/landmark_comparaison.png'
            plt.savefig(save_path)

    
    def create_mask_comparaison_graph(self, data, num_landmarks=5 ,num_images=1, show=True, save=False, save_path=None):
        self.model.eval()
        sample = next(iter(data))

        if num_images > len(data):
            num_images = len(data)

        images, landmarks, masks = sample['image'][:num_images], sample['landmarks'][:num_images], sample['mask'][:num_images]
        images, landmarks , masks = images.to(self.device), landmarks.to(self.device), masks.to(self.device)
        mask_pred = self.model(images)
        landmarks_pred = mask_to_landmarks(mask_pred).float()

    
        for j in range(num_images):
            fig = plt.figure(figsize=(10, 10))
            for i in range(0, 3*num_landmarks, 3):
                plt.subplot(num_landmarks, 3, i+1)
                plot_image(images[j])
                plt.plot(landmarks[j, i//3, 0], landmarks[j, i//3,1], marker='o', color='green')
                plt.plot(landmarks_pred[j, i//3, 0], landmarks_pred[j, i//3,1], marker='o', color='red')

                plt.subplot(num_landmarks, 3, i+2)
                plot_image(masks[j, i//3])

                plt.subplot(num_landmarks, 3, i+3)
                plot_image(mask_pred[j, i//3])
            

        if show:
            plt.show()

        if save:
            path = f'{save_path}/mask_comparaison.png'
            plt.savefig(save_path)




