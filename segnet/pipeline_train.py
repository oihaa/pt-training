
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

import numpy as np
import pickle

import camvidreader 

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
    0]

n_pixel = 480*360

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
       lr =[ param_group['lr'] ]
    return lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def do_train(args, model):

    torch.manual_seed(args.seed)

    
    train_loader, validation_loader, test_loader = camvidreader.get_default_datasets(
        args.batch_size,
        args.test_batch_size,
        args.test_batch_size)

    
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    test_metrics = []

    for epoch in range(args.epochs):      
        train(epoch, args, model, train_loader, optimizer)
        metrics = test(epoch, model, args, validation_loader)
        test_metrics.append(metrics)

        if (epoch > 1) and (epoch % 10) == 0:
            lr_rate = get_learning_rate(optimizer)[0]
            lr_rate *= 0.5
            print("Setting learning rate to: ", lr_rate)            
            adjust_learning_rate(optimizer, lr_rate)
    save_metrics(args.result_folder,test_metrics)            
    torch.save(model.state_dict(), args.result_folder + "/" + "segnet")


def train(epoch, args, model, data_loader, optimizer): 
    model.train()
    pid = os.getpid()
    for b_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)        
        loss = F.nll_loss(F.log_softmax(output),
                          target.long(),
                          weight=torch.FloatTensor(weights).cuda(),
                          ignore_index=11)
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
    save_confusion_matrix(args.result_folder, epoch, con_mat)
    correct = correct / n_pixel
    test_loss = test_loss / n_pixel
    test_loss /= len(data_loader.dataset)  
    print('\tTest set: Average loss:  \t{:.4f},\t Accuray: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return (test_loss, 100. * correct / len(data_loader.dataset))



def create_confusion(predictions, targets):
    pred = np.concatenate(predictions, axis=0)
    targ = np.concatenate(targets, axis=0)

    matrix = confusion_matrix(targ.reshape(-1),pred.reshape(-1) )
    
    return matrix

def save_confusion_matrix(folder, id, matrix):
    with open(folder + "/" + '{:03d}'.format(id), "wb") as pf:
        pickle.dump(matrix, pf)
        pf.close()

def save_metrics(folder, metrics):
    with open(folder + "/Test-metrics", "wb") as pf:
        pickle.dump(metrics, pf)
        pf.close()

    

