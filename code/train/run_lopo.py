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
from txlstm_szpool import *
from lopofn import *
from baselines import *

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("lr", help="learning rate", type=float)
#parser.add_argument("maxiter", help="max train epochs", type=int)
parser.add_argument("cvfold", help="main cross val fold", type=int)

args = parser.parse_args()

#lrdict = {'tgcn':1e-04, 'txmlp':1e-04, 'mlplstm':1e-04, 'ctl':1e-04, 'cnnblstm':1e-06}
for m in ['tgcn', 'txmlp', 'mlplstm', 'ctl', 'cnnblstm', 'sztrack']:
    for lr in [1e-04, 1e-05, 1e-06]:
        x = nested_cv_pretrain(data_root= '/home/dshama1/data-avenka14/deeksha/tuh/',
                       modelname = m,
                       mn_fn='tuh_single_windowed_manifest.csv',
                       cvfold = args.cvfold,
                       cv_root='/home/dshama1/data-avenka14/deeksha/miccai23/',
                       maxiter = 30, lr = lr, 
                       valsize = 25,  
                       use_cuda=True)
