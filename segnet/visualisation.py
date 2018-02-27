import os
import PIL
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from camvidreader import CamvidReader

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]

Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]

Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]

Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]

Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

label_names = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence",
               "Car", "Pedestrian", "Bicyclist", "Unlabelled"]

norm = [104.00699, 116.66877, 122.67892]

def convert_target(img):
    img_arr = np.array(img)
    img_shape = img_arr.shape
    im2arr = img_arr.reshape(-1)
    iii = []        
    for i in range(len(im2arr)):
        iii.append(label_colours[im2arr[i]])
    a = np.array(iii, dtype=np.uint8).reshape(img_shape[0], img_shape[1],3)
    return PIL.Image.fromarray(a)

    

def setup():
    
    train_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="train",
                     transform=transforms.Compose([                      
                         transforms.ToTensor(),
                         transforms.Normalize((104.00699, 116.66877, 122.67892),(1,1,1))       
                     ])
        ),
        batch_size=1, shuffle=False, num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="test",
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((104.00699, 116.66877, 122.67892),(1,1,1))       
                      ])
        ),
        batch_size=1, shuffle=False, num_workers=1)

    return train_loader, test_loader


if __name__ == '__main__':
    
    train_loader, test_loader = setup()

    data_loader = test_loader

    plt.ion()
    for b_idx, (data, target) in enumerate(data_loader):
        img = data.numpy()
        img = img.reshape(img.shape[1],img.shape[2],img.shape[3])
        img = img.transpose(1,2,0)

        img = img + norm
        
        tar = target.numpy()
        tar = tar.reshape(tar.shape[1],tar.shape[2])
        tar = convert_target(tar)        


        plt.figure(1)

        gridspec.GridSpec(10,10)

        ax = plt.subplot2grid((10, 10), (0,0), colspan = 8, rowspan=5)
        ax.set_title("Original")
        plt.imshow(img, vmin=0, vmax=255)
        
        
        ax = plt.subplot2grid((10, 10), (5,0), colspan = 8, rowspan=5)        
        ax.set_title("Segmentet")        
        plt.imshow(tar, vmin=0, vmax=255)

        ax = plt.subplot2grid((10, 10), (0,8 ), colspan = 2, rowspan=10)
        ax.patch.set_alpha(0)        
        ax.axis('off')
        ax.set_xlim(0,6)
        ax.set_ylim(-1,12)
        for t in range(len(label_colours)):
            ax.plot(1, t, 'o', color = np.array(label_colours[t])/256, markersize=15)
            ax.text(2, t, label_names[t], fontsize=10, color='black')

            
        plt.tight_layout()
            
        plt.show(block=False)

        key = input("Next (Enter) or Exit(q)")
        if key == "q":
            break       
    plt.close()
    
    
