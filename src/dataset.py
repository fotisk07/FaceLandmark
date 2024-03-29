import os
import torch
import pandas as pd
from skimage import io, transform, filters
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}
    
class ReduceLandmarks(object):
    def __init__(self, step):
        self.step = step

    def __call__(self, sample):

        landmarks = sample['landmarks']

        landmarks = landmarks[::int(self.step)]


        return {'image': sample['image'], 'landmarks': landmarks}
    
class Normalise(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.astype(np.float32)
        image /= np.max(image)

        return {'image': image, 'landmarks': landmarks}

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir,train=True, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        if train:
            self.landmarks_frame = self.landmarks_frame[:int(len(self.landmarks_frame)*0.8)]
        else:
            self.landmarks_frame = self.landmarks_frame[int(len(self.landmarks_frame)*0.8):]

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0] + ".jpg")
        
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)

        
        sample = {'image': image, 'landmarks': landmarks}
     
        if self.transform:
            sample = self.transform(sample)


        landmarks = sample["landmarks"]
        # Create the mask now that the landmarks are reduced
        mask = np.zeros((len(landmarks), image.shape[0], image.shape[1]), dtype = "float32")
        
        for i in range(landmarks.shape[0]):
            point = landmarks[i]
            if 0 < point[1] < image.shape[0] and 0< point[0] < image.shape[1]:
                mask[i , int(point[1]), int(point[0])] = 1

        mask = filters.gaussian(mask, sigma=10, channel_axis=0)
        mask = mask / np.max(mask)
        sample["mask"] = torch.from_numpy(mask)

        return sample
    


def create_data(csv_file, root_dir, batch_size=32, num_workers=1,validate=False,
                transform=None, reduce_landmarks=20, only_valid=False):
    
    if not transform:
        trans = transforms.Compose([ReduceLandmarks(reduce_landmarks), Normalise(), ToTensor()])


    train_data = FaceLandmarksDataset(csv_file=csv_file,root_dir=root_dir,train=True,
                                      transform=trans)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers)

    valid_data = FaceLandmarksDataset(csv_file=csv_file,root_dir=root_dir,train=False,
                                    transform=trans)

    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    if only_valid:
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
        train_loader = None
    

    return train_loader, valid_loader

