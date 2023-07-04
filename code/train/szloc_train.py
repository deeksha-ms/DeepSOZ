

##from baselines import *
from utils import *
from szloc_loader import *
from szloc import *
from szloc_loss import *
import numpy as np
import pandas as pd
import json
import os

import torch
import torch.nn as nn
import sklearn.metrics as metrics



def cvszloc_finetune(data_root, modelname, mn_fn, cvfold, cv_root='crossval/',maxiter = 50, lr = 1e-04,
                use_cuda=True, pooltype='szpool'):

        device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'
        
        classification_loss = nn.CrossEntropyLoss()  
        map_loss_pos = MapLossL2PosSum(scale=True)
        map_loss_pos = map_loss_pos.to(device)
        map_loss_neg = MapLossL2Neg(scale=True)
        map_loss_neg = map_loss_neg.to(device)
        map_loss_margin = MapLossMargin()
        map_loss_margin = map_loss_margin.to(device)

        chn_sz_weight = 1
        tot_sz_weight = 1
        attn_map_weight_pos = chn_map_weight_pos = 2
        attn_map_weight_neg = chn_map_weight_neg = 1
    
        attn_map_weight_margin = chn_map_weight_margin = 1

        
        manifest = read_manifest(data_root+'data/'+mn_fn, ',')
        save_loc = cv_root+'fold'+str(cvfold)+'/szloc_final/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        train_pts = np.load(cv_root+'fold'+str(cvfold)+'/'+'pts_train'+str(cvfold)+'.npy')
       
        print("\nStarting on fold ", cvfold, save_loc)#, len(val_pts), len(train_pts))

        train_set =  szlocLoader(data_root, train_pts, manifest,
                                    addNoise=False, input_mask=None, normalize=True,
                                    ablate=False, permute=False)
        #val_set =  pretrainLoader(data_root, val_pts, manifest,
        #                          addNoise=False, input_mask=None, normalize=True,
        #                          ablate=False, permute=False)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        #validation_loader = DataLoader(val_set, batch_size=1, shuffle=True)
        model = ctg_11_8(transformer_dropout = 0.15, cnn_dropout=0.15)
        model.to(device)
        model = model.double()
        
        savename = modelname+'_'+str(cvfold)+'_'+str(lr)

        
        optimizer = torch.optim.Adam( model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                B, Nsz, _, _, _ = inputs.shape
                onset_map = data['onset map'].to(device)
                onset_map = onset_map.repeat(B*Nsz, 1)
                labels = data['sz_labels'].long().to(device).reshape(B*Nsz, -1)
                    

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                
                (
                        chn_sz_pred, sz_pred,
                        attn_onset_map, chn_onset_map
                    ) = model(inputs)

                sz_label_idx = list(range(15)) + list(range(30,45))
                chn_sz_loss = chn_sz_weight * classification_loss(
                        chn_sz_pred[:, sz_label_idx, :].transpose(1, 2),
                        labels[:, list(range(15)) + list(range(30,45))])
                tot_sz_loss = tot_sz_weight * classification_loss(
                        sz_pred[:,list(range(15)) + list(range(30,45)), :].transpose(1, 2),
                        labels[:, list(range(15)) + list(range(30,45))])
                total_loss = chn_sz_loss + tot_sz_loss

                attn_map_loss_pos = attn_map_weight_pos * \
                        map_loss_pos(attn_onset_map, onset_map)
                attn_map_loss_neg = attn_map_weight_neg * \
                        map_loss_neg(attn_onset_map, onset_map)
                attn_map_loss_margin = attn_map_weight_margin * \
                        map_loss_margin(attn_onset_map, onset_map)
                    
                    
                total_loss += attn_map_loss_pos + attn_map_loss_neg + attn_map_loss_margin 
                chn_map_loss_pos = chn_map_weight_pos * \
                        map_loss_pos(chn_onset_map, onset_map)
                chn_map_loss_neg = chn_map_weight_neg * \
                        map_loss_neg(chn_onset_map, onset_map)
                chn_map_loss_margin = chn_map_weight_margin * \
                        map_loss_margin(chn_onset_map, onset_map)
              
            
                loss = total_loss + chn_map_loss_pos + chn_map_loss_neg + chn_map_loss_margin
                
                
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()#*y.size(0)
                del labels, onset_map
                
         
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

