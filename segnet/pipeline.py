
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

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
    7.3614]

n_pixel = 480*360

def do_train(rank, par, model):

    torch.manual_seed(par["seed"] + rank)
    
    train_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="train",
                     transform=transforms.Compose([                      
                         transforms.ToTensor(),
                         transforms.Normalize((104.00699, 116.66877, 122.67892),(1,1,1))       
                     ])
        ),
        batch_size=par["batch_size"], shuffle=True, num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        CamvidReader(folder="CamVid", mode="test",
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((104.00699, 116.66877, 122.67892),(1,1,1))       
                      ])
        ),
        batch_size=par["test_batch_size"], shuffle=True, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(par["epochs"]):      
        train(epoch, par, model, train_loader, optimizer)
        torch.save(model.state_dict(), "segnet-" + str(epoch))
        test(model, test_loader)
        

def train(epoch, par, model, data_loader, optimizer): 
    model.train()
    pid = os.getpid()
    for b_idx, (data, target) in enumerate(data_loader):
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
    

def test(model, data_loader): 
    model.eval()
    test_loss = 0
    correct = 0    
    for idx, (data, target) in enumerate(data_loader):        
        data, target = Variable(data, volatile=True), Variable(target).long()    
        output = F.log_softmax(model(data))
        test_loss += F.nll_loss(output, target.long(), size_average=False).data[0] #sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probalility
        correct += pred.eq(target.data).cpu().sum()
        
    correct = correct / n_pixel
    test_loss = test_loss / n_pixel
    test_loss /= len(data_loader.dataset)            
    print('\tTest set: Average loss:  {:.4f}, Accuray: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

    

