import torch 
import matplotlib.pyplot as plt
import numpy 
#from lopofn import * 
##from visualization import * 
import torch.nn as nn
from utils import *
from dataloader import *
import sklearn.metrics as metrics 


class sztrack(nn.Module):
    #gives window wise prediction. No correlation with other windows considered
    def __init__(self, nchns=1):
        super(sztrack, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3, padding=1)

        

        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)
        
        self.drop2 = nn.Dropout(p=0.2)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.lstm = nn.LSTM(input_size=20, hidden_size=20, num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc1 = nn.Linear(40, 2)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, Nsz, T, C, L = x.size()

        
        x = x.view(b*Nsz*T*C,1, L)
       
        x = self.bn1(F.leaky_relu((self.conv1(x)) ))
        x = F.max_pool1d(x, 2)
        x1 = self.drop1(x)
        
        x2 = self.bn2(F.leaky_relu((self.conv2(x1)) ))
        #x2 = F.max_pool1d(x2, 2)
        x2 = self.drop1(x2)
        

        x3 = self.bn2(F.leaky_relu((self.conv3(x2)) ))
        #x3 = F.max_pool1d(x3, 2)
        x3 = self.drop1(x3)
        x3 = x3+x2
 
        x3 = torch.mean(x3, 2) #return (batch, 40)
        hc = x3.view(b, Nsz, T, C, -1)

        rin = hc.transpose(2, 3).reshape(b*Nsz*C, T, -1)
        rout, (hn, cn) = self.lstm(rin)

        rout = rout.reshape(-1, 40)
        rout = self.fc1(rout).reshape(-1, 19, 2)
        #return self.Sigmoid(x)
        maxid = torch.argmax(rout, -2)
        finout = rout[np.arange(rout.shape[0]), maxid[:, 1]]
        #print(b, Nsz, T, C, '\n', hc.shape, rin.shape,maxid.shape, finout.shape)
        
        #print(b, Nsz, T, C, '\n', hc.shape, rin.shape,maxid.shape finout.shape) 
        a = None
        return finout.reshape(b, Nsz, T, -1), hc, a

def build_neighbors(connections_fn):
    """Build the A matrix from the list of channel labels
    """
    adjacent_nodes = {}
    labels2chns = {}
    chns2labels = {}
    label_list = []
    curr_chn = 0
    with open(connections_fn, 'r') as adj_file:
        for line in adj_file:
            # Get adjacencies
            node, neighbors = line.strip().split(':')
            node = node.upper()
            neighbors = neighbors.upper()
            label_list.append(node)
            adjacent_nodes[node] = neighbors.split(',')
            # Keep a map of labels to channels
            chns2labels[curr_chn] = node
            labels2chns[node] = curr_chn
            curr_chn += 1
    nnodes = curr_chn

    # Generate the adjacency matrix and aunt list
    neighbors = []
    A = np.zeros((nnodes, nnodes))
    for chn, label in chns2labels.items():
        neighbors.append([])
        for neighbor in adjacent_nodes[label]:
            kk = labels2chns[neighbor]
            A[chn, kk] = 1
            neighbors[-1].append(kk)
    return neighbors


class STC_layer(nn.Module):
    
    def __init__(self, input_features, output_features, neighbors):
        super().__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        self.neighbors = neighbors

        
        self.tcnn = torch.nn.Conv1d(input_features, output_features, 
                                    3, padding=1)
        self.bn = torch.nn.BatchNorm1d(output_features)
        self.tcnn_comb = torch.nn.Conv1d(2*output_features, output_features,
                                         1, padding=0)
        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        """Perform STC layer

        Args:
            x (B, C, L, T): [description]
        """
        B, C, L, T = x.shape
        
        # Peform TCNN
        h = self.tcnn(x.reshape(B*C, L, T))            # (B*C, OF, T)
        h = self.bn(h)
        h = h.view(B, C, self.output_features, T)   # (B*C, OF, T)
        
        # Perform aggregation over neighbors
        z = h.new(h.shape)
        for cc in range(C):
            neighbors = h[:, self.neighbors[cc], :, :]
            a = torch.max(neighbors, dim=1)[0]
            z[:, cc, :, :] = a
        h = torch.cat((h, z), dim=2)                # (B, C, 2*OF, T)
        h = self.leakyrelu(h)
        h = h.view(B*C, 2*self.output_features, T)
        h = self.leakyrelu(self.tcnn_comb(h))       # (B, C, OF, T)
        h = h.view(B, C, self.output_features, T)
        return h


class TGCN(nn.Module):
    
    def __init__(self, connections_fn=""):
        super().__init__()
        
        self.neighbors ={0: [1,2,3,4],
                  1: [0,4,5,6],
                  2: [0,3,4,6,7,8],
                  3: [0,2,4,5,8,9],
                  4: [0,1,3,5,9],
                  5: [1,3,4,6,9,10],
                  6: [1,2,4,5,10,11],
                  7: [2,8,11,12,13],
                  8: [2,3,4,7,9,10,12,13,14],
                  9: [3,4,5,8,10,13,14,15],
                 10: [4,5,6,8,9,11,14,15,16],
                 11: [6, 7,10, 15, 16],
                 12: [7, 8, 13,16, 17],
                 13: [7, 8, 9, 12, 14,15, 17],
                 14: [8,9,10,13,15,17,18],
                 15: [9,10,11,13,14,16,18],
                 16: [10,11,12,15,18],
                 17: [12,13,14,18],
                 18: [14,15, 16, 17]}

        # Block 1
        self.stc1 = STC_layer(200, 32, self.neighbors)
        self.stc2 = STC_layer(32, 32, self.neighbors)
        
        # Block 2
        self.stc3 = STC_layer(32, 64, self.neighbors)
        self.stc4 = STC_layer(64, 64, self.neighbors)
        
        # Block 3
        self.stc5 = STC_layer(64, 128, self.neighbors)
        self.stc6 = STC_layer(128, 128, self.neighbors)
        
        # Block 4
        self.stc7 = STC_layer(128, 256, self.neighbors)
        self.stc8 = STC_layer(256, 256, self.neighbors)
        
        # Classification
        self.linear1 = torch.nn.Linear(256, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 2)
        
    
    def forward_pass(self, x):
        B, Nsz, T, C, L = x.shape
        x = x.reshape(B*Nsz, T, C, L)

        h = x.transpose(1, 2)                   # (B, C, T, L)
        h = h.transpose(2, 3)                   # (B, C, L, T)
        # Peform TCNN
        h_c = self.stc1(h)
        h_c = self.stc2(h_c)
        h_c = self.stc3(h_c)
        h_c = self.stc4(h_c)
        h_c = self.stc5(h_c)
        h_c = self.stc6(h_c)
        h_c = self.stc7(h_c)
        h_c = self.stc8(h_c)
        
        # Final Classification
        #average pooling across spatial dimension
        
        h_m = h_c.mean(1)
        h_m = h_m.reshape(B*Nsz, -1, T)
        h_m = h_m.transpose(1, 2)                       # (B, T, 256)
        h_m = self.linear1(h_m)                          # (B, T, 512)        
        h_m = self.linear2(h_m)                          # (B,  T, 512)
        h_m = self.linear3(h_m)                          # (B, T, 2)

        # Reshaping
        #h = h.transpose(1, 2)                       # (B, T, C, 2)    
        
        sat = None
        return h_m.reshape(B, Nsz, T, -1) , h_c.reshape(B, Nsz, T, -1), sat

    def forward(self, x):
        y, h, sat = self.forward_pass(x)        
        return y, h, sat
    
class mlp_lstm(nn.Module):
    def __init__(self, cnn_dropout=0.0, gru_dropout=0.0, transformer_dropout=0.0, device='cpu', return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        self.device=device
        # Channel 
        
        
        self.l1 = nn.Linear(200, 100)
        self.l2 = nn.Linear(100, 1)
        self.l3 = nn.Linear(19, 16)
        self.l4 = nn.Linear(16, 32)
        
        # Multichannel GRU
        self.nhidden_sz = 16
        self.multi_lstm = nn.LSTM(input_size=32, hidden_size=self.nhidden_sz,
                                 batch_first=True, bidirectional=True, num_layers=1,
                                 dropout=gru_dropout)

        # Linear layers
        self.multi_linear = nn.Linear(2 * self.nhidden_sz, 2)

    def forward_pass(self, x):
        B, Nsz, T, C, L = x.size()
        
        x = x.reshape(-1, L)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        x = x.reshape(-1, C)
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        
        # Apply multi GRU
        h_m = x.reshape(-1, T, 32)
        self.multi_lstm.flatten_parameters()
        h_m, _ = self.multi_lstm(h_m)     # (B, T, nchannels=40)

        h_m = self.multi_linear(h_m.reshape(B*Nsz*T, -1 ))
        
        sat = None
        h_c = None
            
        return h_m.reshape(B,Nsz,T, -1), h_c, sat #h_c, h_m, a


    def forward(self, x):
        proba, h_c, a = self.forward_pass(x)
        return proba, h_c, a

class cnn_transformer_lstm(nn.Module):
    def __init__(self, cnn_dropout=0.0, gru_dropout=0.0, transformer_dropout=0.0, device='cpu', return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        # Channel encoder components
        self.nchn_c = 200
        self.pos_encoder = nn.Embedding(20, 32).to(device)
        self.device=device
        # Channel Transformer

        self.cnn1 = nn.Conv1d(1, 32, 5, 5)
        self.cnn2 = nn.Conv1d(32, 32, 5, 2)
        self.tx_encoder = nn.TransformerEncoderLayer(32, nhead=8, dim_feedforward=128, batch_first=True)

        # Multichannel GRU
        self.nhidden_sz = 16
        self.multi_lstm = nn.LSTM(input_size=32, hidden_size=self.nhidden_sz,
                                 batch_first=True, bidirectional=True, num_layers=1,
                                 dropout=gru_dropout)

        # Linear layers
        self.multi_linear = nn.Linear(2 * self.nhidden_sz, 2)
        chn_neighbors = {0: [1,2,3,4],
                  1: [0,4,5,6],
                  2: [0,3,4,6,7,8],
                  3: [0,2,4,5,8,9],
                  4: [0,1,3,5,9],
                  5: [1,3,4,6,9,10],
                  6: [1,2,4,5,10,11],
                  7: [2,8,11,12,13],
                  8: [2,3,4,7,9,10,12,13,14],
                  9: [3,4,5,8,10,13,14,15],
                 10: [4,5,6,8,9,11,14,15,16],
                 11: [6, 7,10, 15, 16],
                 12: [7, 8, 13,16, 17],
                 13: [7, 8, 9, 12, 14,15, 17],
                 14: [8,9,10,13,15,17,18],
                 15: [9,10,11,13,14,16,18],
                 16: [10,11,12,15,18],
                 17: [12,13,14,18],
                 18: [14,15, 16, 17]}

        self.src_mask = torch.ones(20,20, requires_grad=False).bool().to(device)

        for i in range(19):
            self.src_mask[i, chn_neighbors[i]] = False
            self.src_mask[i,i] = False
        self.src_mask[-1, :] = False

    def forward_pass(self, x):
        B, Nsz, T, C, L = x.size()

        #cnn
        x = x.reshape(B*Nsz*T*C, 1,  L)
        x = F.max_pool1d(F.relu(self.cnn1(x)), 2)
        x = F.max_pool1d(F.relu(self.cnn2(x)), 2)
        x = x.mean(-1)
        x = x.reshape(B, Nsz, T, C, 32)

        # add pos encoding

        chn_pos = torch.arange(19).to(self.device)
        pos_emb = self.pos_encoder(chn_pos)[None, None, :,:]
        h_c = x + pos_emb
        h_m = self.pos_encoder(torch.tensor([19]*B*T*Nsz).view(B, Nsz, T, -1).to(self.device))


        # Apply Transformer
        h_c = h_c.reshape(B*Nsz*T, C, 32)
        tx_input = torch.cat((h_c, h_m.reshape(B*Nsz*T, 1, 32)), dim=1)
        tx_input = self.tx_encoder(tx_input, src_mask=self.src_mask)
        h_c = tx_input[:, :-1, :].view(B*Nsz, T, C, 32)
        h_m = tx_input[:, -1, :].view(B*Nsz, T, -1)



        # Apply multi GRU
        self.multi_lstm.flatten_parameters()
        h_m, _ = self.multi_lstm(h_m)     # (B, T, nchannels=40)

        h_m = self.multi_linear(h_m.reshape(B*Nsz*T, -1 ))

        if self.return_attn:
            _, amat= self.tx_encoder.self_attn(tx_input, tx_input, tx_input, attn_mask=self.src_mask)
            attnmap = amat[:, -1,:-1]
            max_chn_across_time = []
            for i in range(B*Nsz*T):
                max_chn_across_time.append(torch.argmax(attnmap[i, :]))
            temp_neighbours = (torch.argmax(h_m, -1) == 1).reshape(-1).long().detach()
            max_chn_across_time = torch.tensor(max_chn_across_time)
            onset, x = np.histogram(max_chn_across_time[temp_neighbours], bins=19, range=(0,20))
            sat = torch.tensor(onset/onset.max()).repeat(Nsz, 1)
        else:
            sat = None

        return h_m.reshape(B,Nsz,T, -1), h_c.reshape(B, Nsz, T, C, -1), sat #h_c, h_m, a

    def forward(self, x):
        proba, h_c, a = self.forward_pass(x)
        return proba, h_c, a

class transformer_mlp(nn.Module):
    def __init__(self, cnn_dropout=0.0, gru_dropout=0.0, transformer_dropout=0.0, device='cpu', return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        # Channel encoder components
        self.nchn_c = 200
        self.pos_encoder = nn.Embedding(20, 200).to(device)
        self.device=device
        # Channel Transformer

        self.tx_encoder = nn.TransformerEncoderLayer(200, nhead=8, dim_feedforward=256, batch_first=True)


        # Linear layers
        self.multi_linear = nn.Linear(200, 2)
        chn_neighbors = {0: [1,2,3,4],
                  1: [0,4,5,6],
                  2: [0,3,4,6,7,8],
                  3: [0,2,4,5,8,9],
                  4: [0,1,3,5,9],
                  5: [1,3,4,6,9,10],
                  6: [1,2,4,5,10,11],
                  7: [2,8,11,12,13],
                  8: [2,3,4,7,9,10,12,13,14],
                  9: [3,4,5,8,10,13,14,15],
                 10: [4,5,6,8,9,11,14,15,16],
                 11: [6, 7,10, 15, 16],
                 12: [7, 8, 13,16, 17],
                 13: [7, 8, 9, 12, 14,15, 17],
                 14: [8,9,10,13,15,17,18],
                 15: [9,10,11,13,14,16,18],
                 16: [10,11,12,15,18],
                 17: [12,13,14,18],
                 18: [14,15, 16, 17]}

        self.src_mask = torch.ones(20,20, requires_grad=False).bool().to(device)

        for i in range(19):
            self.src_mask[i, chn_neighbors[i]] = False
            self.src_mask[i,i] = False
        self.src_mask[-1, :] = False
    
    def forward_pass(self, x):
        B, Nsz, T, C, L = x.size()

        # add pos encoding
        chn_pos = torch.arange(19).to(self.device)
        pos_emb = self.pos_encoder(chn_pos)[None, None, :,:]
        h_c = x + pos_emb
        h_m = self.pos_encoder(torch.tensor([19]*B*T*Nsz).view(B, Nsz, T, -1).to(self.device))
        
      
        # Apply Transformer
        h_c = h_c.reshape(B*Nsz*T, C, 200)
        tx_input = torch.cat((h_c, h_m.reshape(B*Nsz*T, 1, 200)), dim=1)
        tx_input = self.tx_encoder(tx_input, src_mask=self.src_mask)
        h_c = tx_input[:, :-1, :].view(B*Nsz, T, C, 200)
        h_m = tx_input[:, -1, :].view(B*Nsz, T, -1)


        h_m = self.multi_linear(h_m.reshape(B*Nsz*T, -1 ))
        
        if self.return_attn:
            _, amat= self.tx_encoder.self_attn(tx_input, tx_input, tx_input, attn_mask=self.src_mask)
            attnmap = amat[:, -1,:-1]
            max_chn_across_time = []
            for i in range(B*Nsz*T):      
                max_chn_across_time.append(torch.argmax(attnmap[i, :]))
            temp_neighbours = (torch.argmax(h_m, -1) == 1).reshape(-1).long().detach()
            max_chn_across_time = torch.tensor(max_chn_across_time)
            onset, x = np.histogram(max_chn_across_time[temp_neighbours], bins=19, range=(0,20))
            sat = torch.tensor(onset/onset.max()).repeat(Nsz, 1)
        else:
            sat = None
            
        return h_m.reshape(B,Nsz,T, -1), h_c.reshape(B, Nsz, T, C, -1), sat #h_c, h_m, a


    def forward(self, x):
        proba, h_c, a = self.forward_pass(x)
        return proba, h_c, a
    
class CNN_BLSTM(nn.Module):
    #gives window wise prediction. No correlation with other windows considered
    def __init__(self, nchns=19):
        super(CNN_BLSTM, self).__init__()
        self.conv5a = nn.Conv1d(nchns, 5, kernel_size=3, padding=1)
        self.conv5b = nn.Conv1d(5, 5, kernel_size=3, padding=1)
        self.conv10a = nn.Conv1d(5, 10, kernel_size=3, padding=1)
        self.conv10b = nn.Conv1d(10, 10, kernel_size=3, padding=1)
        self.conv20a = nn.Conv1d(10, 20, kernel_size=3, padding=1)
        self.conv20b = nn.Conv1d(20, 20, kernel_size=3, padding=1)
        self.conv40a = nn.Conv1d(20, 40, kernel_size=3, padding=1)
        self.conv40b = nn.Conv1d(40, 40, kernel_size=3, padding=1)
        

        self.bn5a = nn.BatchNorm1d(5)
        self.bn5b = nn.BatchNorm1d(5)
        self.bn10a = nn.BatchNorm1d(10)
        self.bn10b = nn.BatchNorm1d(10)
        self.bn20a = nn.BatchNorm1d(20)
        self.bn20b = nn.BatchNorm1d(20)
        self.bn40a = nn.BatchNorm1d(40)
        self.bn40b = nn.BatchNorm1d(40)
        
        self.drop2 = nn.Dropout(p=0.2)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=20, num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc1 = nn.Linear(40, 2)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, Nsz, T, C, L = x.size()

        
        x = x.view(b*Nsz*T, C, L)
       
        x = self.bn5a(F.leaky_relu((self.conv5a(x)) ))
        x = F.max_pool1d(self.bn5b(F.leaky_relu((self.conv5b(x)) )), 2)
        x = self.drop1(x)
        
        x = self.bn10a(F.leaky_relu((self.conv10a(x)) ))
        x = F.max_pool1d(self.bn10b(F.leaky_relu((self.conv10b(x)) )), 2)
        x = self.drop1(x)
        
        x = self.bn20a(F.leaky_relu((self.conv20a(x)) ))
        x = F.max_pool1d(self.bn20b(F.leaky_relu((self.conv20b(x)) )), 2)
        x = self.drop1(x)
        
        x = self.bn40a(F.leaky_relu((self.conv40a(x)) ))
        x = F.max_pool1d(self.bn40b(F.leaky_relu((self.conv40b(x)) )), 2)
        x = self.drop2(x)
 
        x = torch.mean(x, 2) #return (batch, 40)
        x = x.view(b*Nsz, T, 40)
        
        rout, (hn, cn) = self.lstm(x)
       
        rout = rout.reshape(-1, 40)
        rout = self.fc1(rout)
        #return self.Sigmoid(x)
        hc = None
        a = None
        return rout.reshape(b, Nsz, T, -1), hc, a
    

class sztrack(nn.Module):
    #gives window wise prediction. No correlation with other windows considered
    def __init__(self, nchns=1):
        super(sztrack, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3, padding=1) 
    
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)

        self.drop2 = nn.Dropout(p=0.2)
        self.drop1 = nn.Dropout(p=0.1)

        self.lstm = nn.LSTM(input_size=20, hidden_size=20, num_layers=2, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(40, 2)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, Nsz, T, C, L = x.size()


        x = x.view(b*Nsz*T*C,1, L)

        x = self.bn1(F.leaky_relu((self.conv1(x)) ))
        x = F.max_pool1d(x, 2)
        x1 = self.drop1(x)

        x2 = self.bn2(F.leaky_relu((self.conv2(x1)) ))
        #x2 = F.max_pool1d(x2, 2)
        x2 = self.drop1(x2)


        x3 = self.bn2(F.leaky_relu((self.conv3(x2)) ))
        #x3 = F.max_pool1d(x3, 2)
        x3 = self.drop1(x3)
        x3 = x3+x2

        x3 = torch.mean(x3, 2) #return (batch, 40)
        x3 = x3.view(b, Nsz, T, C, -1)

        rin = x3.transpose(2, 3).reshape(b*Nsz*C, T, -1)
        rout, (hn, cn) = self.lstm(rin)

        rout = rout.reshape(-1, 40)
        rout = self.fc1(rout).reshape(-1, 19, 2)
        #return self.Sigmoid(x)
        hc = rout.reshape(b*Nsz, T, 19, 2)
        maxid = torch.argmax(rout, -2)
        finout = rout[np.arange(rout.shape[0]), maxid[:, 1]]
        #print(b, Nsz, T, C, '\n', hc.shape, rin.shape,maxid.shape, finout.shape)
        
        #print(b, Nsz, T, C, '\n', hc.shape, rin.shape,maxid.shape finout.shape) 
        a = None
        return finout.reshape(b, Nsz, T, -1), hc, a
