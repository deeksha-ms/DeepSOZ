import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import math
from utils import *

###from tuh_loader import *
###from visualization import *

class pretrainLoader(Dataset):
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
        if self.addNoise:
            n, t, c, d = X.shape
            xhat = X.reshape(-1, 19, 200).copy()
            #noise_ind = []
            noise_labels = np.zeros((n*t, c))
            for item in range(n*t):
                noisy_c = np.random.choice(nonmask_ind, 1) #just one noisy channel out of the 10
                xhat[item, noisy_c,:] = np.random.normal(0,self.sigma, d)
                if len(mask_ind)!=0:
                    xhat[item, mask_ind,:] = np.zeros((len(mask_ind), d))
                #noise_ind.append(noisy_c)
                noise_labels[item, noisy_c] = 1
        
            xhat = xhat.reshape(n, t, c, d)
            noise_labels = noise_labels.reshape(n, t, c)
            
        if self.permute:
            permute_type = 0#np.random.choice(5)
            if permute_type==0: #swap R-L
                indices = np.array([1, 0, 6, 5, 4, 3,2,11,10,9,8,7,16,15,14,13,12,18,17])
            elif permute_type==1: #swap hemis
                indices = np.array([17, 18, 12, 13, 14, 15, 15, 7,8,9,10,11,2,3,4,5,6,0,1])
            elif permute_type==2:
                indices = np.array([18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
            elif permute_type==3:
                indices = np.random.permutation(np.arange(19))
            else:
                indices = np.arange(19)
            
            indices = indices.reshape(-1)
            X = X[:, :, indices, :]
            soz = soz.reshape(-1)[indices]
            if xhat!=[]:
                xhat = xhat[:, :, indices, :]
            

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
    
    def get_input_mask(self, soz):
        soz_ind = list(np.where(soz==1)[0])
        neigb_ind = []
        neigb_ind = list(set(np.concatenate([neigb_ind+self.chn_neighbours[i] for i in soz_ind])))
        neigb_ind = [i for i in neigb_ind if i not in soz_ind]
        soz_len = len(soz_ind)
        neigb_len = len(neigb_ind)
        max_len = self.nchn -self.maxMask
        if (soz_len + neigb_len ) >= max_len:
            mask_ind = np.concatenate((soz_ind, 
                                       np.array([np.random.choice(neigb_ind, 
                                                                  max_len-soz_len, 
                                                                  replace=False)]).reshape(-1)))
  
        else:
            rand_ind = np.array([np.random.choice([i for i in range(19) if (i not in soz_ind and i not in neigb_ind)],
                                                  max_len-soz_len-neigb_len,
                                                  replace=False)] ).reshape(-1)
        
            mask_ind = np.concatenate((soz_ind, 
                                       neigb_ind, 
                                       rand_ind))
            
        
        return mask_ind
        
    def add_noise_soz(self, xorg, soz):
        soz_ind = list(np.where(soz==1)[0])
        Nsz, T, C, d = xorg.shape
        xorg = xorg.reshape(-1, C, d)
        xmax = xorg.max()
        xmin = xorg.min()
        X = xorg.copy()
        noise_models = ['normal', 'uniform',  'gamma', 'rayleigh', 'binomial'] #poisson
        for t in range(Nsz*T):
            noise_type =  np.random.choice(5)
            if noise_type==0:
                replace_chns = np.random.normal(0, 1, (len(soz_ind), d))
            elif noise_type==1:
                replace_chns = np.random.uniform(xmin, xmax, (len(soz_ind), d))
            elif noise_type==2:
                replace_chns = np.random.gamma(1, 1, (len(soz_ind), d))
            elif noise_type==3:
                replace_chns = np.random.rayleigh(2, (len(soz_ind), d))

            else:
                speckle_loc = np.random.binomial(1, 0.5, (len(soz_ind), d))
                speckle = xorg[t, soz_ind, :]
                speckle[speckle_loc == 1] = 0
                X[t, soz_ind] = speckle
                
            if noise_type!=4:
                X[t, soz_ind] = replace_chns
                
            
            
        return X.reshape(Nsz, T, C, d)
