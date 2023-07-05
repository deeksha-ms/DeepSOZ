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
##import matplotlib.pyplot as plt

##import pyedflib
'''
def readEDF(filename, 
            root_dir = '',
           req_chns = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']):
    
    f = pyedflib.EdfReader(root_dir+filename)
    n = f.signals_in_file
    data = []
    for i in range(n):
        if str(f.signal_label(i)).split(" ")[0].split("'")[1] in req_chns:
            data.append(f.readSignal(i))
        
    return np.array(data)
'''
def read_manifest(filename, d=';'):
   
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=d)
        dicts = list(reader)
    return dicts

def write_manifest(new_manifest, fname='untitled_mn.csv', d=';'):
    with open(fname, 'w') as csvfile:
        fieldnames = new_manifest[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

        writer.writeheader()
        for i in range(len(new_manifest)):
            writer.writerow(new_manifest[i] )
            
            
def crop_points(mn, crop_len=300):
    dur = json.loads(mn['duration'])
    #print((mn['sz_starts']))
    sz_start = np.array(json.loads(mn['sz_starts']))
    sz_end = np.array(json.loads(mn['sz_ends']))
    nsz = json.loads(mn['nsz'])
    szlen = sz_end - sz_start 
    ep = [(sz_start[i+1]+sz_end[i])//2  for i in range(0, nsz-1, 1)]
    ep = ep + [dur]
    bp = [0] + ep[:-1]

    s = np.zeros(nsz)
    e = np.zeros(nsz)

    for i in range(nsz):
        halflen = (crop_len - szlen[i])//2
        
        s[i] = int(max(bp[i], sz_start[i]-halflen))
        e[i] = int(min(ep[i], s[i]+crop_len))

    return s, e

        
def get_ptlist(n=15):
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'O', 'Q', 'S']


def apply_croppad(mn, savePath, crop_len=600):

    nsz = json.loads(mn['nsz'])
    tot_dur = json.loads(mn['duration'])
    fn = mn['fn'].split('.')[0]
    xfeat = np.load(root+'/Xwindows/'+fn+'_win.npy')
    y = np.load(root+'/Ywindows/'+fn+'_label.npy')
    s,e = crop_points(mn, crop_len)
    #print(nsz, tot_dur, newdur, s, e)
    
    for i in range(nsz):
        xnew = xfeat[ int(s[i]): int(e[i]), :, :]
        ynew = y[int(s[i]):int(e[i])]
        newdur = len(ynew)
        if newdur<crop_len:
            xnew = np.pad(xnew, ( (int(crop_len-newdur), 0), (0,0), (0,0) ), 'constant' )
            ynew = np.pad(ynew, ( int(crop_len-newdur), 0), 'constant')
        #print(np.isnan(xnew).any())
        #plt.plot(ynew)
        #plt.show()
        np.save(savePath+fn+'_win_'+str(i)+'.npy', xnew)
        np.save(savePath+fn+'_label_'+str(i)+'.npy', ynew)

def lopoCrop(pt_list, manifest,root, fol = '/cropped600/'):
    for pt in pt_list:
        #print(pt)
        test_mn = list(filter(lambda mn:mn['fn'].startswith( 'E_' + pt + "_"), manifest))
        os.mkdir(root+pt+fol)
        for mn_item in test_mn:        
            apply_croppad(mn_item, root+pt+fol, 600)
    
    for n_fold, test_pt in enumerate(pt_list):
        if not os.path.isdir(root+test_pt+'/train'):
            os.mkdir(root+test_pt+'/train')
        #if not os.path.isdir(root+test_pt+'/test'):
         #    os.mkdir(root+test_pt+'/test')
    
        train_pt =   pt_list[:n_fold] + pt_list[n_fold+1:]
        #test_mn_list = list(filter(lambda mn:mn['fn'].startswith( 'E_' + test_pt + "_"), manifest))
        print(test_pt)
        shutil.copytree(root+test_pt+fol, root+test_pt+'/test')
         
        for pt in train_pt:
            files = os.listdir(root+pt+fol)
            for fn in files:
                shutil.copy(root+pt+fol+fn, root+test_pt+'/train/'+fn)
    return

