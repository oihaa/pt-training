import os
import PIL
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt

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

        ax = plt.subplot(1,2,1)        
        plt.imshow(img, vmin=0, vmax=255)
        
        ax = plt.subplot(1,2,2)        
        plt.imshow(tar, vmin=0, vmax=255)        

        plt.show(block=False)

        key = input("Next or Esc")
        if key == "Q":
            break       
    plt.close()
    
    
def dont_2():    

    for img_file in os.listdir("CamVid/testannot"):
        print(img_file)

        
        img = PIL.Image.open("CamVid/testannot/" + img_file)
        img = convert_target(img)

        plt.figure
        plt.imshow(img, vmin=0, vmax=255)
        plt.show(block=False)

        key = input("Next or Esc")

        plt.close()

        if key == "Q":
            break


        
        
        
def dont():
        img = cv2.imread("CamVid/testannot/" + img_file)

        dstimage = np.zeros(img.shape, dtype=np.uint8)

        ea = np.zeros([244,3])
        ea2 = np.concatenate((label_colours,ea), axis=0)
        ea3 = np.reshape(ea2,(1,256,3))


        
        cv2.LUT(img, ea3 , dstimage)

        print(img)
        print("Dill")
        print(dstimage)
        print("asd")
        print(ea3)

        cv2.imshow('image', dstimage)

        k = cv2.waitKey(0)


