import os
import collections
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import numpy as np
import imageio

class CamvidReader(Dataset):

    def __init__(self, folder, mode, transform=False, target_transform=False):

        self.folder = folder
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        modes = ["train", "test", "val"]

        self.file_list = {}

        for sub_folder in modes:
            self.file_list[sub_folder] = os.listdir(folder + "/" + sub_folder)


    def __len__(self):
        return(len(self.file_list[self.mode]))

    def __getitem__(self, index):
        file_name = self.file_list[self.mode][index]
        
        img = imageio.imread(self.folder + "/" + self.mode + "/" + file_name)
        img = np.array(img, dtype=np.uint8)

        label = imageio.imread(self.folder + "/" + self.mode + "annot/" + file_name)
        label = np.array(label, dtype=np.uint8)

        
        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            label = self.target_transform(label)



        return img, label, file_name    


def get_default_datasets(batch_size, val_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="train",
                     transform=transforms.Compose([                      
                         transforms.ToTensor()
                     ])
        ),
        batch_size=batch_size, shuffle=True, num_workers=1)

    validation_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="val",
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])
        ),
        batch_size=val_batch_size, shuffle=True, num_workers=1)
    
    test_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="test",
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])
        ),
        batch_size=test_batch_size, shuffle=True, num_workers=1)

    return train_loader, validation_loader, test_loader
