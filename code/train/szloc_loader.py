
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import math
from utils import *


class szlocLoader(Dataset):
    def __init__(self,root,  ptlist, manifest, 
                 normalize=True, transform=None, 
                 maxMask=10, sigma=0.2, maxSeiz = 10, addNoise=False, input_mask=False, ablate=False, permute=False):
        self.ptlist = ptlist
        self.root= root
        self.mnlist = [mnitem for mnitem in manifest if json.loads(mnitem['pt_id']) in ptlist ]
        self.transform = transform
        self.normalize = normalize
        self.maxMask = maxMask
        self.nchn = 19
        self.sigma = sigma
        self.maxSeiz = maxSeiz 
        self.input_mask= input_mask
        self.addNoise = addNoise
        self.ablate = ablate
        self.permute = permute
        self.chn_neighbours = {0: [1,2,3,4], 
                  1: [0,4,5,6], 
                  2: [0,3,7,8], 
                  3: [0,2,4,8,9], 
                  4: [0,1,3,5,9], 
                  5: [1,4,6,9,10],
                  6: [1,5,10,11], 
                  7: [2,8,12,13,17], 
                  8: [2,3,7,9,12,13,14], 
                  9: [3,4,5,8,10,13,14,15], 
                 10: [5,6,9,11,14,15,16], 
                 11: [6, 10, 15, 16, 18], 
                 12: [7, 8, 13, 17], 
                 13: [7, 8, 9, 12, 14, 17],
                 14: [8,9,10,13,15,17,18],
                 15: [9,10,11,14,16,18], 
                 16: [10,11,15,18], 
                 17: [7,12,13,14,18], 
                 18: [11, 14,15, 16, 17]}

    def __getitem__(self, idx):

        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        sz_starts = np.ceil(json.loads(mnitem['sz_starts']))
        sz_ends = np.ceil(json.loads(mnitem['sz_ends']))
        xloc = self.root+mnitem['loc']
        yloc =  xloc.split('.')[0] + '_label.npy'

        X = np.load(xloc)[:self.maxSeiz, :,:,:]
        Y = np.load(yloc)[:self.maxSeiz]
        soz = self.load_onset_map(mnitem)
        
        szt = Y.argmax(1)
        nsz = Y.shape[0]
        X_ = []
        Y_ = []
        for j in range(nsz):
            
            x = np.random.randint(1, 15, 1)[0]
            s = max(0, szt[j]-x-15)
            e = s+45
            if e>600:
                e=600
                s=e-45
            X_.append(X[j, s:e])
            Y_.append(Y[j, s:e])
        X = np.array(X_)
        Y = np.array(Y_)
        if self.normalize:
            X = (X - np.mean(X))/np.std(X)
            
        if self.ablate:
            X = self.add_noise_soz(X, soz)
            
        
        if self.input_mask:            
            nonmask_ind = self.get_input_mask(soz)
            mask_ind = [i for i in range(19) if i not in nonmask_ind]
        else:
            nonmask_ind = np.arange(19)
            mask_ind = []
        #noise_ind = np.random.choice(mask_ind, self.noiseNum)
        #noise_label = np.zeros(self.nchn)

        
        noise_labels =  []
        xhat = []

        return {'patient numbers': fn,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
                'onset map':torch.Tensor(soz), #soz
                'noisy_buffers': torch.Tensor(xhat), #masked modified
                'mask_ind': torch.Tensor(mask_ind).long(), 
                'noise_labels': torch.Tensor(noise_labels)
               }
   
    
    def load_onset_map(self, mnitem):
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 
                    'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        for i,chn in enumerate(req_chn):
            if mnitem[chn] != '':
                soz[i] = json.loads(mnitem[chn])
           
        return soz
    
    def __len__(self):                       #gives number of recordings
        return len(self.mnlist)

        

