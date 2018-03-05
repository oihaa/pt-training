
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import pickle

import camvidreader 


n_pixel = 480*360


def calculate(data_loader):
    targets = []
    for b_idx, (data, target) in enumerate(data_loader):
        targets.append(target.numpy())

    result = np.concatenate(targets, axis=0)
    result = result.reshape(-1)

    bincount = np.bincount(result)
    print(bincount)

    sum = len(result)

    bincount = bincount /sum
    return bincount

def adv_calculate(data_loader):

    tot = []

    for b_idx, (data, target) in enumerate(data_loader):
        if b_idx == 0:
            bins = np.bincount(target.numpy().reshape(-1))
        else:    
            bins = np.append(bins,np.bincount(target.numpy().reshape(-1)), axis=0)

    bins = bins.reshape(-1, 12)
    tot = np.sum(bins,axis=0)
    nz = np.count_nonzero(bins, axis=0)
    freq = tot/(nz*n_pixel)

    median = np.median(freq)

    weights = median/freq

    #print(tot)
    #print(nz)
    #print(freq)
    #print(weights)

    return weights
        

if __name__ == '__main__':
   
    train_loader, validation_loader, test_loader = camvidreader.get_default_datasets(1, 1, 1)


    train_results = adv_calculate(train_loader)
    print("test")
    print(train_results)

    validation_results = adv_calculate(validation_loader)
    print("validation")
    print(validation_results)

    test_results = adv_calculate(test_loader)
    print("test")      
    print(test_results)


    
    


