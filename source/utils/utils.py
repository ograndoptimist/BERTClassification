import copy
import torch.nn as nn


def clones(module, N):
    """
        Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
