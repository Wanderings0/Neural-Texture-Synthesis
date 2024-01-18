"""
Author: Xiaohui Zhang
calculate gram matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.data import Batch
from tqdm import tqdm
import time, sys

from VGG19 import VGG19, modifiedVGG19, get_vgg19_model

def gram_mat(input: list[torch.Tensor]):
    """
    param input: [batch_size, channel, height, width]
    return: [batch_size, channel, channel]
    """
    gram_mat = []
    for feat in input:
        channel, height, width = feat.size()
        input = input.view(channel, height * width)
        input_t = input.transpose()
        gram = torch.mm(input, input_t)
        gram_mat.append(gram)
    return gram_mat

def extract_feat(model):
    assert isinstance(model, VGG19) or isinstance(model, modifiedVGG19)

    feat_maps = model.features_maps
    list_feat = list(feat_maps.values())
    list_gram = gram_mat(list_feat)

    return list_gram


