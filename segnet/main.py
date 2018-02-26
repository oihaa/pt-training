
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from pipeline import do_train
from segnet import SegNet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segnet training')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.batch_size = 10
    args.test_batch_size = 10
    args.epochs = 60
    args.seed = 1

        
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(1)

    model = SegNet(3,13)
    model.initialized_with_pretrained_weights()
    model = nn.DataParallel(model)
    
    if args.cuda:
        model.cuda()
    
    model.share_memory() 
    do_train(1, args, model)
