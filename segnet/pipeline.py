
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

from camvidreader import CamvidReader

#    weight_by_label_freqs: true
#    ignore_label: 11

weights = [
    0.2595,
    0.1826,
    4.5640,
    0.1417,
    0.9051,
    0.3826,
    9.6446,
    1.8418,
    0.6823,
    6.2478,
    7.3614,
    1,00,
    1.00,]

n_pixel = 480*360

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def do_train(rank, args, model):

    plt.ion()
    torch.manual_seed(args.seed + rank)
    
    train_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="train",
                     transform=transforms.Compose([                      
                         transforms.ToTensor()#,
#                         transforms.Normalize((104.00699, 116.66877, 122.67892),(1,1,1))       
                     ])
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    validation_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="val",
                      transform=transforms.Compose([
                          transforms.ToTensor()#,
#                          transforms.Normalize((104.00699, 116.66877, 122.67892),(1,1,1))       
                      ])
        ),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

#    scheduler = optim.lr_scheduer.StepLR(optimizer, step_size=8, gamma=0.5)

    
    for epoch in range(args.epochs):      
        train(epoch, args, model, train_loader, optimizer)
        test(epoch, model, args, validation_loader)

        if (epoch > 1) and (epoch % 8) == 0:
            lr_rate = get_learning_rate(optimizer)[0]
            lr_rate *= 0.5
            print("Setting learning rate to: ", lr_rate)            
            adjust_learning_rate(optimizer, lr_rate)
    torch.save(model.state_dict(), "segnet" + str(epoch))


def train(epoch, args, model, data_loader, optimizer): 
    model.train()
    pid = os.getpid()
    for b_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)        
        loss = F.nll_loss(F.log_softmax(output), target.long())
        loss.backward()
        optimizer.step()
        if b_idx % 10 == 0:
            print('{}\tTrain Epoch: {}[{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
            pid, epoch, b_idx *len(data), len(data_loader.dataset),
            100. * b_idx / len(data_loader), loss.data[0]))
    

def test(epoch, model, args, data_loader): 
    model.eval()
    test_loss = 0
    correct = 0
    first =  True
    predictions = []
    targets = []

    for idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        data, target = Variable(data, volatile=True), Variable(target).long()    
        output = F.log_softmax(model(data))
        test_loss += F.nll_loss(output, target.long(), size_average=False).data[0] #sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probalility
        correct += pred.eq(target.data).cpu().sum()

        predictions.append(output.data.max(1, keepdim=True)[1].cpu().numpy())
        targets.append(target.data.cpu().numpy())
        
    con_mat = create_confusion(predictions, targets)
    save_confusion_matrix(epoch, con_mat)
    correct = correct / n_pixel
    test_loss = test_loss / n_pixel
    test_loss /= len(data_loader.dataset)  
    print('\tTest set: Average loss:  \t{:.4f},\t Accuray: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))



def create_confusion(predictions, targets):
    pred = np.concatenate(predictions, axis=0)
    targ = np.concatenate(targets, axis=0)

    matrix = confusion_matrix(targ.reshape(-1),pred.reshape(-1) )
    print(matrix)
    return matrix

def save_confusion_matrix(id, matrix):
    with open("results/" + '{:03d}'.format(id), "wb") as pf:
        pickle.dump(matrix, pf)
        pf.close()

    

