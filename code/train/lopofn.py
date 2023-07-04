from txlstm_szpool import *
from utils import *
from dataloader import *
from baselines import *
import numpy as np
import pandas as pd
import json
import os

import torch
import torch.nn as nn
import sklearn.metrics as metrics



def nested_cv_pretrain(data_root, modelname, mn_fn, cvfold, cv_root='crossval/',maxiter = 30, lr = 1e-05, valsize = 20,
                use_cuda=False):

    device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'

    detloss = nn.CrossEntropyLoss(weight = torch.Tensor([0.2, 0.8]).double().to(device))
    manifest = read_manifest(data_root+'data/'+mn_fn, ',')
    save_loc = cv_root+'fold'+str(cvfold)+'/baselines/'
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    pt_list = np.load(cv_root+'fold'+str(cvfold)+'/'+'pts_train'+str(cvfold)+'.npy')
    step = valsize
    subfold_range = int(100/valsize)
    for fold in range(subfold_range):

        val_pts = pt_list[fold*step: (fold+1)*step]
        train_pts = [pt for pt in pt_list if pt not in val_pts ]
        print("\nStarting on fold ", fold, len(val_pts), len(train_pts))

        train_set =  pretrainLoader(data_root, train_pts, manifest,
                                    addNoise=False, input_mask=None, normalize=True,
                                    ablate=False, permute=False)
        val_set =  pretrainLoader(data_root, val_pts, manifest,
                                  addNoise=False, input_mask=None, normalize=True,
                                  ablate=False, permute=False)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        validation_loader = DataLoader(val_set, batch_size=1, shuffle=True)

        if modelname == 'cnnblstm':
                model = CNN_BLSTM()
        elif modelname == 'ctl':
                model = cnn_transformer_lstm(device=device)
        elif modelname == 'txmlp':
                model = transformer_mlp(device=device)
        elif modelname == 'mlplstm':
                model = mlp_lstm(device=device)
        elif modelname == 'tgcn':
                model  = TGCN()
        else:
            model = transformer_lstm(transformer_dropout=0.15, device=device)
        
        savename = modelname+'_pretrained_cv'+str(cvfold)+'_'+str(lr)+'_'
        model.double()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        for epoch in range(1, maxiter+1):

            epoch_loss = 0.0
            train_len = len(train_loader)
            for batch_idx, data in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = data['buffers']
                det_labels = data['sz_labels'].long().to(device)

                Nsz = inputs.shape[1]

                inputs = inputs.to(torch.DoubleTensor()).to(device)
                k_pred, hc, _  = model(inputs)
                del inputs
                loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()#*y.size(0)
                del det_labels
                if batch_idx%50==0:
                    print('Epoch: ', epoch, ' batch id: ', batch_idx, 'Loss: ', loss.item())

            epoch_val_loss = 0.
            val_len = len(validation_loader)
            for batch_idx,data in enumerate(validation_loader):

                optimizer.zero_grad()
                with torch.no_grad():
                    optimizer.zero_grad()
                    inputs = data['buffers']
                    det_labels = data['sz_labels'].long().to(device)
                    Nsz = inputs.shape[1]

                    inputs = inputs.to(torch.DoubleTensor()).to(device)
                    k_pred, psoz, ysoz  = model(inputs)
                    del inputs
                    loss = detloss(k_pred.reshape(-1,2), det_labels.reshape(-1))
                    epoch_val_loss += loss
                    
            epoch_loss = epoch_loss/train_len
            epoch_val_loss = epoch_val_loss/val_len
            train_losses.append(epoch_loss)
            val_losses.append(epoch_val_loss)

            #if epoch_loss <= train_losses[-1] and epoch_val_loss <= val_losses[-1]:
            #torch.save(model.state_dict(), '/home/deeksha/EEG_Sz/GenProc/results/lstmAE_'+pt+str(ch_id)+'.pth.tar')        
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, epoch_loss, epoch_val_loss))

            torch.save(model.state_dict(), save_loc+savename+ str(fold)+'.pth.tar')
        del model, optimizer, train_loader, validation_loader
    return None
