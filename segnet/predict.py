
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from pipeline_predict import do_predict
from segnet import SegNet#Maybe with dropout


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segnet training')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--batch_size', action='store', dest='batch_size',
                        default=10, help='Size of minibatch')
    parser.add_argument('--test_batch_size', action='store', dest='test_batch_size',
                        default=10, help='Size of test minibatch')
    parser.add_argument('--seed', action='store', dest='seed',
                        default=1, help='Pytorch seed')
    parser.add_argument('--model', action='store', dest='model',
                        default='results/segnet', help='Name of the model file')
    parser.add_argument('--result-folder', action='store', dest='result-folder',
                        default='results', help='Where to store result files')


    args = parser.parse_args()

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

        
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(1)

    model = SegNet(3,12)

    model.initialized_with_pretrained_weights()
    
    #print(model.state_dict().keys())

    state_dict = torch.load(args.model)
    #print(state_dict.keys())



    model = nn.DataParallel(model)
    
    if args.cuda:
        model.cuda()
    
    model.share_memory()
    model.load_state_dict(state_dict)    
    do_predict(args, model)
