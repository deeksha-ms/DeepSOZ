import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt





class MapLossL1Pos(torch.nn.Module):
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            maxes, _ = torch.max(onset_map_pred, dim=0)
            onset_map_pred = onset_map_pred / (maxes + torch.tensor(1e-6))
        pos_loc_max, _ = torch.max(onset_map_pred * onset_map, dim=1)
        return torch.mean(1 - pos_loc_max)


class MapLossL1PosSum(torch.nn.Module):
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            maxes, _ = torch.max(onset_map_pred, dim=0)
            onset_map_pred = onset_map_pred / (maxes + torch.tensor(1e-6))
        pos_loc_sum = torch.sum(onset_map_pred * onset_map, dim=1)
        if self.scale:
            factor = torch.sum(onset_map, dim=1)
            pos_loc_sum = pos_loc_sum / factor
        return torch.mean(1 - pos_loc_sum)


class MapLossL2PosSum(torch.nn.Module):
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + torch.tensor(1e-6))
        pos_loc_sum = torch.sum((onset_map - onset_map_pred * onset_map) ** 2, dim=1)
        if self.scale:
            factor = torch.sum(onset_map, dim=1)
            pos_loc_sum = pos_loc_sum / factor
        return torch.mean(pos_loc_sum)


class MapLossL2PosMax(torch.nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + torch.tensor(1e-6))
        pos_loc_max, _ = torch.max((onset_map - onset_map_pred * onset_map) ** 2, dim=1)
        return torch.mean(pos_loc_max)


# class MapLossL1Neg(torch.nn.Module):
#     def __init__(self, normalize=True, scale=True):
#         super().__init__()
#         self.normalize = normalize
#         self.scale = scale
    
#     def forward(self, onset_map_pred, onset_map):
#         if self.normalize:
#             maxes, _ = torch.max(onset_map_pred, dim=0)
#             onset_map_pred = onset_map_pred / (maxes + torch.tensor(1e-6))
#         neg_loc_sum = torch.sum(onset_map_pred * (1 - onset_map), dim=1)
#         if self.scale:
#             factor = torch.sum(1 - onset_map, dim=1)
#             neg_loc_sum = neg_loc_sum / factor
#         return torch.mean(neg_loc_sum)


class MapLossL2Neg(torch.nn.Module):
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + torch.tensor(1e-6))
        neg_loc_sum = torch.sum((onset_map_pred * (1 - onset_map)) ** 2, dim=1)
        if self.scale:
            factor = torch.sum(1 - onset_map, dim=1)
            neg_loc_sum = neg_loc_sum / factor
        return torch.mean(neg_loc_sum)


class MapLossMargin(torch.nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + torch.tensor(1e-6))
        pos_loc_max, _ = torch.max(onset_map_pred * onset_map, dim=1)
        neg_loc_max, _ = torch.max(onset_map_pred * (1 - onset_map), dim=1)
        return torch.mean((1 - pos_loc_max ** 2 + neg_loc_max  ** 2) / 2)


class MapLossL2(torch.nn.Module):
    def __init__(self, normalize=True, scale=True):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
    
    def forward(self, onset_map_pred, onset_map):
        if self.normalize:
            B, C = onset_map_pred.shape
            maxes, _ = torch.max(onset_map_pred, dim=1)
            onset_map_pred = onset_map_pred / (maxes.view(B, 1) + torch.tensor(1e-6))
        neg_loc_sum = torch.sum((onset_map_pred * (1 - onset_map)) ** 2, dim=1) / C
        pos_loc_sum = torch.sum((onset_map - onset_map_pred * onset_map) ** 2, dim=1) / C
        return torch.mean(neg_loc_sum + pos_loc_sum)
