import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import combinations

import numpy as np


def get_loss(loss_name, loss_config=None):
    if loss_name == 'BCE':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name == 'BCE_weighted':
        pos_weight = torch.ones([1])*loss_config.pos_weight
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise Exception('Unknown loss type')
    return loss


