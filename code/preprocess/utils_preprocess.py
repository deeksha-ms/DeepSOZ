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


def readEDF(filename, ns,
            root_dir = '',
           req_chns = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','CZ','C4','T4','T5','P3','PZ','P4','T6','O1','O2'], 
           ):
    
    f = pyedflib.EdfReader(root_dir+filename)
    n = f.signals_in_file
    data =  {key: None for key in req_chns}
    fs = {key: None for key in req_chns}
    for i in range(n):
        
        chn = str(f.signal_label(i)).split(" ")[1].split("-")[0]
        if chn in req_chns:
           
            data[chn] = f.readSignal(i)                
            fs[chn] = f.getSampleFrequencies()[i]

    f.close()
    for chn in req_chns:
        if fs[chn] == None:                
                data[chn] = np.zeros(ns)
                fs[chn] = fs['FP1']
      
    data = np.array(list(data.values()))
    fs = np.array(list(fs.values()))
    #print(data.shape, fs)
    return data, fs

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

def resample_eeg(x, fs, ns, fsnew=200.):
    
    return sig.resample(x, int(np.float(ns)*fsnew/fs))

def resample_manifest(mnitem, fs, ns, fsnew=200.):
    nsnew = int(np.float(ns)*fsnew/fs)
    mnitem_new = mnitem.copy()
    mnitem_new['ns'] = nsnew
            
def crop_points(mn, crop_len=600):
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

#preprocess
def applyLowPass(x, fs, fc=30, N=4):
    """Apply a low-pass filter to the signal
    """
    wc = fc / (fs / 2)
    b, a = scipy.signal.butter(N, wc)
    return scipy.signal.filtfilt(b, a, x, method='gust')


def applyHighPass(x, fs, fc=1.6, N=4):
    """Apply a high-pass filter to the signal
    """
    wc = fc / (fs / 2)
    b, a = scipy.signal.butter(N, wc, btype='highpass')
    return scipy.signal.filtfilt(b, a, x, method='gust')

def clip(x, clip_level):
    """Clip the signal to a given standard deviation"""
    mean = np.mean(x)
    std = np.std(x)
    return np.clip(x, mean - clip_level * std, mean + clip_level * std)

def preprocess(x, fs, f2=30, f1=1.6, N=4, clip_level=2):
    x_fil = applyHighPass(applyLowPass(x, fs, f2, N), fs, f1, N)
    return clip(x_fil, clip_level)

def compute_nwindows(duration, window_length, overlap):

    advance = window_length - overlap
    nwindows = int(np.floor((duration - window_length) / advance)) + 1
    #return np.arange(0, np.floor(fs*duration), np.floor(fs*advance))
    return nwindows
    
def create_label(duration, sz_starts, sz_ends, window_length, overlap,
                 post_sz_state=False):

    nwindows = compute_nwindows(duration, window_length, overlap)
    # Initialize all labels to 0
    label = np.zeros(nwindows)#, dtype=np.long)
    '''
    #this might be wrong
    '''
    window_time = (window_length - overlap) * np.arange(nwindows)
    if post_sz_state:
        label[np.where(window_time >= sz_ends[0])] = 2
        # Set any labels between starts to 1
    for start, end in zip(sz_starts, sz_ends):
        start = np.round(start, 1)
        end = np.round(end, 1)
        label[np.where((window_time >= start) * (window_time <= end))] = 1
        
    return label

def applyWindows(dataSample, fs, duration, window_length, overlap):
    
    nchns = dataSample.shape[0]
    nwindows = compute_nwindows(duration, window_length, overlap)
    startArray = (window_length-overlap)*fs*np.arange(nwindows)
    endArray = startArray + fs*window_length     
    startArray = startArray.astype(int)
    endArray = endArray.astype(int)
    #windowedSample = np.zeros(nwindows[i], int(manifestAll[i]['nchns'], fs*window_len ))
    windowedSample = np.array([[dataSample[nc, startArray[nwin]:endArray[nwin]] for nc in range(nchns)] for nwin in range(nwindows)])
    #    windowedAll.append(windowedSample)
            
    return windowedSample


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

