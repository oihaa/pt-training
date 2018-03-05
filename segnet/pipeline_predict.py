
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
import imageio

def do_predict(args, model):

    torch.manual_seed(args.seed)

    
    train_loader, validation_loader, test_loader = camvidreader.get_default_datasets(
        args.batch_size,
        args.test_batch_size,
        args.test_batch_size)

    
    predict(args, model, test_loader)


    

def predict(args, model, data_loader): 
    model.eval()

    for idx, (data, target, filename) in enumerate(data_loader):
        if args.cuda:
            data = data.cuda()
        
        data = Variable(data, volatile=True)   
        output = F.log_softmax(model(data))

        pred = output.data.max(1)[1] # get the index of the max log-probalility

        pz = pred.size()

        for i in range(pz[0]):
            print(filename[i])
            imageio.imwrite("predict/"+filename[i], pred[i,:,:].cpu().numpy())


        
    

