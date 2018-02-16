
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from pipeline import do_train
from segnet import SegNet


if __name__ == '__main__':

    par = {'batch_size' : 2,
           'test_batch_size' : 10,
           'epochs' : 10,
           'learning_rate' : 0.01,
           'momentum' : 0.5,
           'seed' : 1,
           'processes' : 2}

    torch.manual_seed(par['seed'])

    model = SegNet(3,13)
    model.share_memory() 
    model.initialized_with_pretrained_weights()
    
    processes = []
    for rank in range(1):
        p = mp.Process(target=do_train, args=(rank, par, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
