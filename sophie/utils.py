import os
import math
import random
import numpy as np

import torch

from constants import *

def get_dset_path(dset_name, dset_type):
    return os.path.join('datasets', dset_name, dset_type)

def relative_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def bce_loss(pred, target):
    neg_abs = -pred.abs()
    loss = pred.clamp(min=0) - pred * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.9, 1)
    # return bce_loss(scores_fake, y_fake)
    return torch.nn.BCELoss()(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.8, 0.99)
    y_fake = 1 - y_real  #torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = torch.nn.BCELoss()(scores_real, y_real)  #bce_loss(scores_real, y_real)
    loss_fake = torch.nn.BCELoss()(scores_fake, y_fake)  #bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def l2_loss(pred_traj, pred_traj_gt, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / (seq_len * batch)
    elif mode == 'raw':
        return torch.sqrt(loss.sum(dim=2)+1e-6).sum(dim=1)

def displacement_error(pred_traj, pred_traj_gt, future_valid, mode='mean'):
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = (loss**2)[future_valid > 0]
    loss = torch.sqrt(loss.sum(dim=1)) #.sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    if mode == 'mean':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss

def displacement_error_by_time(pred_traj, pred_traj_gt, future_valid, time=8, mode='mean'):
    n_steps = time/0.5
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = (loss**2)
    loss = torch.sqrt(loss[:, :int(n_steps)].sum(dim=2))[future_valid[:, int(n_steps)-1] > 0] #.sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    if mode == 'mean':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(
    pred_pos, pred_pos_gt, future_valid, mode='sum'
):
    loss = (pred_pos_gt - pred_pos)[future_valid[:,-1]>0]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.mean(loss)
