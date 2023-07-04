from txlstm_szpool_new import *
from baselines import *
from utils import *
from dataloader import *

from szloc_loader import *
import numpy as np
import pandas as pd
import json
import os

import torch
import torch.nn as nn
import sklearn.metrics as metrics
def loc_pred(hc, proba):
    B, T, C, _ = hc.shape
    p = F.softmax(proba.reshape(-1, 2), -1)[:, 1].reshape(B, T)
    delp = p[:, 1:] - p[:, :-1]
    delp[delp<0] = 0
    hc_pred = F.softmax(hc, -1)[:, :, :, 1].reshape(-1, T, C)
    pred_chn = torch.sum(delp.reshape(-1, 44, 1)*hc_pred[:, :-1, :], 1)
    pred_chn/= (pred_chn.sum()+1e-6)
    
    return pred_chn.reshape(B,  C)


def map_loss(hc,pdet, true_onset):
    predchn = loc_pred(hc.reshape(-1, 45, 19, 2), pdet)
    device= 'cuda:0'
    B, C = predchn.shape
    right = np.array([1, 5, 6, 10, 11, 15, 16, 18]).reshape(1, -1)
    left = np.array([0, 2,3,7,8,12,13,17]).reshape(1, -1)
    ant = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(1, -1)
    post = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).reshape(1, -1)
    
    predchn = predchn.reshape(-1, 19)
    true_onset.reshape(-1, 19)
    true_right = 1 if (true_onset[0][right]==1).any() else 0
    true_ant = 1 if (true_onset[0][ant]==1).any() else 0
    true_hemi = torch.tensor([true_right]*B).long().to(device)
    true_reg = torch.tensor([true_ant]*B).long().to(device)
    
    loss = 0
    bce = nn.CrossEntropyLoss()
    
    pred_right  = predchn[:, right].sum(-1)
    pred_left = predchn[:, left].sum(-1)
    pred_hemi = torch.hstack((pred_left, pred_right))
    pred_ant = predchn[:,ant].sum(-1)
    pred_post = predchn[:,post].sum(-1)
    pred_reg = torch.hstack((pred_post, pred_ant))
    
    return bce(pred_hemi, true_hemi) + bce(pred_reg, true_reg)


def cv_finetune(data_root, modelname, mn_fn, cvfold, cv_root='crossval/',maxiter = 30, lr = 1e-05,
                use_cuda=True, pooltype='szpool'):

        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        sozloss = nn.BCELoss()
        detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))  
        l1 = nn.L1Loss()

        
        manifest = read_manifest(data_root+'data/'+mn_fn, ',')
        save_loc = cv_root+'fold'+str(cvfold)+'/models/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        pts = np.load(cv_root+'fold'+str(cvfold)+'/'+'pts_train'+str(cvfold)+'.npy')
        #val_pts = np.load(cv_root+'fold'+str(cvfold)+'/'+'pts_test'+str(cvfold)+'.npy')
        train_pts = pts
        #val_pts = pts[75:]
        print("\nStarting on fold ", cvfold)#, len(val_pts), len(train_pts))

        train_set =  szlocLoader(data_root, train_pts, manifest,
                                    addNoise=False, input_mask=None, normalize=True,
                                    ablate=False, permute=False)
        #val_set =  pretrainLoader(data_root, val_pts, manifest,
        #                          addNoise=False, input_mask=None, normalize=True,
        #                          ablate=False, permute=False)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        #validation_loader = DataLoader(val_set, batch_size=1, shuffle=True)
        #if modelname =='txmlp':
        #    pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-04) + '_' +'.pth.tar'
        #    model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        #elif modelname =='ctl':
        #    pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-04) + '_' +'.pth.tar'
        #    model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        #elif modelname =='tgcn':
        #    pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-04) + '_' +'.pth.tar'
        #    model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        #elif modelname =='txlstm_nomask':
        #    pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(1e-05) + '_' +'.pth.tar'
        #    model = txlstm_szpool(transformer_dropout=0.2,  pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        #else:
        #    prelrs = [1e-5, 1e-5, 1e-5, 1e-4, 1e-4]
        #    pretrainedfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(prelrs[cvfold]) + '_' +'.pth.tar'
        #    model = txlstm_szpool(transformer_dropout=0.15, pretrained=pretrainedfn, device=device, modelname=modelname, pooltype=pooltype)
        model = sztrack()
        pretrainfn = save_loc+modelname+'_pretrained_cv' +str(cvfold)+'_'+ str(0.001) + '_' +'.pth.tar'
        states = torch.load(pretrainfn)
        model.load_state_dict(states)

        savename = modelname+'_'+pooltype+'_finetuned_cv'+str(cvfold)+'_'+str(lr)
        
        model = model.to(device)
        model.double()
        
        optimizer = torch.optim.Adam( model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                Nsz = inputs.shape[1]
                soz_labels = data['onset map'].to(device)
                soz_labels = soz_labels.repeat(Nsz, 1)
                det_labels = data['sz_labels'].long().to(device)


                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, psoz, z   = model(inputs)
                del inputs
                loss1 = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
                loss2 = map_loss(psoz,k_pred, soz_labels)
                loss = loss1 + loss2 #+ 0.1*l1(psoz, torch.zeros_like(psoz))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()#*y.size(0)
                del det_labels
                if batch_idx%50==0:
                    print('Epoch: ', epoch, ' batch id: ', batch_idx, 'Loss: ', loss.item())
            '''
            epoch_val_loss = 0.
            val_len = len(validation_loader)
            for batch_idx,data in enumerate(validation_loader):

                optimizer.zero_grad()
                with torch.no_grad():
                    optimizer.zero_grad()
                    inputs = data['buffers']
                    Nsz = inputs.shape[1]
                    soz_labels = data['onset map'].to(device)
                    soz_labels = soz_labels.repeat(Nsz, 1)
                    det_labels = data['sz_labels'].long().to(device)
                    

                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, psoz, z, a  = model(inputs)
                    del inputs
                    loss1 = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
                    loss2 = sozloss(psoz.float(), soz_labels) #+ sozloss(ysoz.float(), soz_labels)
                    loss = 0.1*loss1 + loss2 #+ 0.1*l1(psoz, torch.zeros_like(psoz))
                    epoch_val_loss += loss
            '''        
            epoch_loss = epoch_loss/train_len
            #epoch_val_loss = epoch_val_loss/val_len
            train_losses.append(epoch_loss)
            #val_losses.append(epoch_val_loss)

            #if epoch_loss <= train_losses[-1] and epoch_val_loss <= val_losses[-1]:
            #torch.save(model.state_dict(), '/home/deeksha/EEG_Sz/GenProc/results/lstmAE_'+pt+str(ch_id)+'.pth.tar')        
            #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, epoch_loss, epoch_val_loss))

            torch.save(model.state_dict(), save_loc+savename +'.pth.tar')
            
            
        del model, optimizer, train_loader#, validation_loader
        return None        
