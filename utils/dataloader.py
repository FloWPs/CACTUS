import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class WMHDataset(Dataset):
    def __init__(self, data_dir, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.data_dir = data_dir

        class_names = os.listdir(data_dir)
        num_class = len(class_names)
        self.idx_to_class = {0:'false_alarms', 1:'true_alarms'}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
         
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = np.load(image_filepath) 
        image = image.astype(np.float32)

        # # select sagittal and coronal projection & duplicate over 3 channels
        # imageS = cv2.cvtColor(image[:, :, 1], cv2.COLOR_BGR2RGB) # sagittal slice only
        # imageC = cv2.cvtColor(image[:, :, 2], cv2.COLOR_BGR2RGB) # coronal slice only

        # select axial and coronal projection & duplicate over 3 channels
        imageA = np.dstack((image[:, :, 0], image[:, :, 0], image[:, :, 0])) # axial slice only
        imageC = np.dstack((image[:, :, 2], image[:, :, 2], image[:, :, 2])) # coronal slice only
        # convert from float32 to uint8
        imageA = cv2.normalize(imageA, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        imageC = cv2.normalize(imageC, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        # # select axial and sagittal projection & duplicate over 3 channels
        # imageS = cv2.cvtColor(image[:, :, 0], cv2.COLOR_BGR2RGB) # axial slice only
        # imageC = cv2.cvtColor(image[:, :, 1], cv2.COLOR_BGR2RGB) # sagittal slice only
        
        label = image_filepath.split('\\')[1] # MODIF FLORA
        label = self.class_to_idx[label]

        if self.transform is not None:
            imageA = self.transform(imageA)
            imageC = self.transform(imageC)
        
        imageAC = [imageA, imageC]
        imageAC = torch.cat(imageAC, dim=0)

        return imageAC, label