
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from pipeline import do_train

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


if __name__ == '__main__':

    par = {'batch_size' : 64,
           'test_batch_size' : 1000,
           'epochs' : 10,
           'learning_rate' : 0.01,
           'momentum' : 0.5,
           'seed' : 1,
           'processes' : 2}

    torch.manual_seed(par['seed'])

    model = Net()
    model.share_memory() 

    processes = []
    for rank in range(1):
        p = mp.Process(target=do_train, args=(rank, par, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
