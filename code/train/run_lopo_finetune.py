import numpy as np
import torch
import csv
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

from utils import *
from torch import Tensor
from dataloader import *
from txlstm_szpool_new import *
from lopofn_finetune import *
from baselines import *

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("lr", help="learning rate", type=float)
#parser.add_argument("maxiter", help="max train epochs", type=int)
##parser.add_argument("cvfold", help="main cross val fold", type=int)

args = parser.parse_args()
models = ['tgcn', 'ctl', 'txmlp']
pooltypes = ['szpool', 'maxpool']
#for p in pooltypes:
for cvfold in range(0, 1, 15):
            x = cv_finetune(data_root= '/home/dshama1/data-avenka14/deeksha/tuh/',
                      modelname = 'txlstm',
                    mn_fn='tuh_single_windowed_manifest.csv',
                   cvfold = cvfold,
                   cv_root='/home/dshama1/data-avenka14/deeksha/miccai23/',
                   maxiter = 50, lr = args.lr,  
                    use_cuda=True, pooltype='szpool')
'''
for m in models:
    for cvfold in range(0, 15, 1):
           x = cv_finetune(data_root= '/home/dshama1/data-avenka14/deeksha/tuh/',
                     modelname = m,
                    mn_fn='tuh_single_windowed_manifest.csv',
                   cvfold = cvfold,
                  cv_root='/home/dshama1/data-avenka14/deeksha/miccai23/',
                 maxiter = 50, lr = 1e-04,  
                use_cuda=True, pooltype='szpool')
'''
