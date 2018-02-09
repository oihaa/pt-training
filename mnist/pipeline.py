
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from reader import myMnistReader

def do_train(rank, par, model):

    torch.manual_seed(par["seed"] + rank)
    
    train_loader_m = torch.utils.data.DataLoader(
        myMnistReader(mode="Train",
                      transform=transforms.Compose([                      
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))                      
                      ])),
        batch_size=par["batch_size"], shuffle=True, num_workers=1)

    test_loader_m = torch.utils.data.DataLoader(
        myMnistReader(mode="Test",
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))                      
                      ])),
        batch_size=par["test_batch_size"], shuffle=True, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    for epoch in range(par["epochs"]):
        train(epoch, par, model, train_loader_m, optimizer)
        test(model, test_loader_m)
        

def train(epoch, par, model, data_loader, optimizer): 
    model.train()
    pid = os.getpid()
    for b_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if b_idx % 100 == 0:
            print('{}\tTrain Epoch: {}[{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
            pid, epoch, b_idx *len(data), len(data_loader.dataset),
            100. * b_idx / len(data_loader), loss.data[0]))

def test(model, data_loader): 
    model.eval()
    test_loss = 0
    correct = 0    
    for idx, (data, target) in enumerate(data_loader):        
        data, target = Variable(data, volatile=True), Variable(target)    
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] #sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probalility
        correct += pred.eq(target.data).cpu().sum()
        
    test_loss /= len(data_loader.dataset)            
    print('\tTest set: Average loss:  {:.4f}, Accuray: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

