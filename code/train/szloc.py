import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, channels, nlayers, kernel_size=3,
                 stride=1, padding=1,
                 residual=False, batch_norm=False):
        super().__init__()

        self.residual = residual
        self.batch_norm = batch_norm

        self.convs = nn.ModuleList(
            [nn.Conv1d(channels, out_channels=channels, 
                       kernel_size=kernel_size, stride=stride, padding=padding) for ii in range(nlayers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(channels) for ii in range(nlayers)])
        self.relu = nn.LeakyReLU()
        

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        h = self.convs[0](x)
        if self.batch_norm:
            h = self.bns[0](h)
        for ii, conv in enumerate(self.convs[1:]):
            h = self.relu(h)
            h = conv(h)
            if self.batch_norm:
                h = self.bns[ii](h)
        if self.residual:
            h = h + x
        h = self.relu(h)
        return h + x


class ctg_11_8(nn.Module):
    def __init__(self, cnn_dropout=0.0, gru_dropout=0.0, transformer_dropout=0.0):
        super().__init__()

        self.tgt_mask=nn.parameter.Parameter(torch.eye(19),
                                             requires_grad=False)

        # Channel encoder components
        self.nchn_c = 80
        self.ConvEmbeddingC = nn.Conv1d(1, out_channels=10,
                                 kernel_size=7, stride=1, padding=3)
        self.ConvC1 = ConvBlock(10, 1, residual=True, kernel_size=7, padding=3)
        self.ProjC1 = nn.Conv1d(10, 20, kernel_size=1, stride=2, padding=0)
        self.ConvC2 = ConvBlock(20, 1, residual=True, kernel_size=7, padding=3)
        self.ProjC2 = nn.Conv1d(20, 40, kernel_size=1, stride=2, padding=0)
        self.ConvC3 = ConvBlock(40, 1, residual=True, kernel_size=7, padding=3)
        self.ProjC3 = nn.Conv1d(40, 80, kernel_size=1, stride=2, padding=0)
        self.ConvC4 = ConvBlock(80, 1, residual=True, kernel_size=7, padding=3)
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # Multichannel Encoder
        self.nchn_m = 80
        self.ConvEmbeddingM = nn.Conv1d(19, out_channels=40,
                                        kernel_size=7, stride=1, padding=3)
        self.Conv1 = ConvBlock(40, 2, residual=True, kernel_size=7, padding=3)
        self.ProjM1 = nn.Conv1d(40, 80, kernel_size=1, stride=2, padding=0)
        self.Conv2 = ConvBlock(80, 2, residual=True, kernel_size=7, padding=3)
        self.ProjM2 = nn.Conv1d(80, 80, kernel_size=1, stride=2, padding=0)
        self.Conv3 = ConvBlock(80, 2, residual=True, kernel_size=7, padding=3)
        
        # Channel Transformer
        self.channel_transformer = nn.Transformer(80, num_encoder_layers=1,
                                                  num_decoder_layers=1,
                                                  dim_feedforward=128,
                                                  batch_first=True,
                                                  dropout=transformer_dropout)

        # Channel GRU
        self.nhidden_c = 40
        self.channel_gru = nn.GRU(input_size=80, hidden_size=self.nhidden_c,
                            batch_first=True, bidirectional=True, num_layers=2,
                            dropout=gru_dropout) 

        # Multichannel GRU
        self.nhidden_sz = 40
        self.multi_gru = nn.GRU(input_size=80, hidden_size=self.nhidden_sz,
                                 batch_first=True, bidirectional=True, num_layers=1,
                                 dropout=gru_dropout)

        # Linear layers
        self.channel_linear = nn.Linear(2 * self.nhidden_c, 2)
        self.multi_linear = nn.Linear(2 * self.nhidden_sz, 2)
        self.onset_linear = nn.Linear(2 * self.nhidden_sz, 1)
        self.sig = nn.Sigmoid()
    def forward_pass(self, x):
        
        B,Nsz, T, C, L = x.size()
        x = x.reshape(B*Nsz, T, C, L)
        # CNN Layers
        h_c = self.cnn_dropout(self._channel_encoder(x))              # (B, T, C, nchannels=80)
        h_m = self.cnn_dropout(self._multichannel_encoder(x))         # (B, T, nchannels=160)

        # Apply Transformer
        h_c = h_c.reshape(B*Nsz*T, C, 80)
        h_c = self.channel_transformer(torch.cat((h_c, h_m.reshape(B*Nsz*T, 1, 80)), dim=1), h_c)
        h_c = h_c.view(B*Nsz, T, C, 80)

        # Apply channel GRU
        h_c = h_c.transpose(1, 2)                   # (B, C, T, nchannels=80)
        h_c = h_c.reshape(B*Nsz*C, T, 80)               # (B*C, T, nchannels=80)
        self.channel_gru.flatten_parameters()
        h_c, _ = self.channel_gru(h_c)              # (B*C, T, nchannels=80)
        h_c = h_c.view(B*Nsz, C, T, 2 * self.nhidden_c)
        h_c = h_c.transpose(1, 2)

        # Apply multi GRU
        self.multi_gru.flatten_parameters()
        h_m, _ = self.multi_gru(h_m)     # (B, T, nchannels=80)

        # Linear Layers
        h_c = self.channel_linear(h_c)
        a = self.onset_linear(h_m)
        a = torch.softmax(a, dim=1)
        h_m = self.multi_linear(h_m)


        return h_c, h_m, a
    
    def _channel_encoder(self, x):
        """Receives full input and outputs channel encodings

        Args:
            x (tensor): Input data (B, T, C, L)

        Returns:
            tensor: Channel encodings (B, T, C, 20)
        """
        B, T, C, L = x.size()
        h = self.ConvEmbeddingC(x.view(B*T*C, 1, L))
        h = self.ConvC1(h)
        h = self.ProjC1(h)
        h = self.ConvC2(h)
        h = self.ProjC2(h)
        h = self.ConvC3(h)
        h = self.ProjC3(h)
        h = self.ConvC4(h)
        h = torch.mean(h.view(B, T, C, self.nchn_c, -1), dim=4)
        return h
    
    def _multichannel_encoder(self, x):
        """[summary]

        Args:
            x (tensor): Input data (B, T, C, L)

        Returns:
            tensor: Multichannel encodings (B, T, 20)
        """
        B, T, C, L = x.size()
        h = x.view(B*T, C, L)
        h = self.ConvEmbeddingM(h)
        h = self.Conv1(h)
        h = self.ProjM1(h)
        h = self.Conv2(h)
        h = self.ProjM2(h)
        h = self.Conv3(h)
        h = torch.mean(h.view(B, T, self.nchn_m, -1), dim=3)
        return h

    def _attn_onset_map(self, h, a):
        B, T, Channels, classes = h.shape
        probs = F.softmax(h, dim=3)
        onset_map = torch.sum(a.view((B, T, 1))
                              * probs[:, :, :, 1], dim=1)
        return onset_map

    def _channel_onset_map(self, h_c):
        channel_probs = F.softmax(h_c, dim=3)
        max_channel_probs, _ = torch.max(channel_probs[:, :, :, 1], dim=2)
        attn = F.relu(max_channel_probs[:, 1:] - max_channel_probs[:, :-1])
        onset_map = torch.sum(
            attn.unsqueeze(2) * channel_probs[:, 1:, :, 1], dim=1)
        return onset_map

    def _max_channel_logits(self, h):
        B, T, _, _ = h.shape
        probs = F.softmax(h, dim=3)
        dev = h.get_device()
        if dev == -1:
            dev = None
        max_logits = torch.zeros((B, T, 2), device=dev)
        for bb in range(B):
            max_channels = torch.argmax(probs[bb, :, :, 1], dim=1)
            for tt in range(T):
                max_logits[bb, tt, :] = h[bb, tt, max_channels[tt], :]
        return max_logits

    def forward(self, x):
        B, Nsz, T, C, L = x.shape
        h_c, h_m, a = self.forward_pass(x)
        
        channel_sz_logits = self._max_channel_logits(h_c)
        attn_onset_map = self._attn_onset_map(h_c, a)
        chn_onset_map = self._channel_onset_map(h_c)
        return (channel_sz_logits, h_m,
               attn_onset_map, chn_onset_map)

    def predict_proba(self, x):
        """Predict each time point, should only be used for single sequences

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        h_c, h_m, attn = self.forward_pass(x.unsqueeze(0))
        sz_logits = self._max_channel_logits(h_c)
        chn_sz_pred = torch.softmax(sz_logits, dim=2)
        sz_pred = torch.softmax(h_m, dim=2)
        chn_pred = torch.softmax(h_c, dim=3)
        attn_onset_map = self._attn_onset_map(h_c, attn)
        chn_onset_map = self._channel_onset_map(h_c)
        return (sz_pred.squeeze(0), chn_sz_pred.squeeze(0), chn_pred.squeeze(0),
                attn_onset_map.squeeze(0), chn_onset_map.squeeze(0), attn)
