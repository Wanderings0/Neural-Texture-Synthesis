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
from typing import List
import cv2

from VGG19 import VGG19, modifiedVGG19, get_vgg19_model
import torch

def gram_mat(input: List[torch.Tensor]):
    """
    param input: [channel, height, width]
    return: [channel, channel]
    """
    gram_mat = []
    for feat in input:
        channel, height, width = feat.size()
        feat = feat.view(channel, height * width)
        feat_t = feat.transpose(0,1)
        gram = torch.mm(feat, feat_t) / (channel * height * width * 2)
        gram_mat.append(gram)
    return gram_mat

def extract_gram(model):
    assert isinstance(model, VGG19) or isinstance(model, modifiedVGG19)

    feat_maps = model.features_maps
    list_feat = list(feat_maps.values())
    list_gram = gram_mat(list_feat)

    return list_gram


class styleLoss(nn.Module):
    def __init__(self, gt: List[torch.Tensor]):
        super(styleLoss, self).__init__()
        self.gt = gt
        self.loss = 0.0

    def forward(self, input: List[torch.Tensor]):
        assert len(input) == len(self.gt)

        for i, G in enumerate(input):
            self.loss += (G - self.gt[i]).pow(2).sum()

        return self.loss
    

# not finished
def gram_mse(input: List[torch.Tensor], gt: List[torch.Tensor], weight=1.0):
    assert len(input) == len(gt)

    num = len(input)
    loss = []
    grad = []
    for i in range(num):
        in_gram = input[i]
        gt_gram = gt[i]

    pass
#############################################


def generate_white_noise(size):
    image = torch.rand(size)
    image = image * 255
    cv2.imwrite("white_noise.jpg", image.detach().numpy().transpose(1,2,0))
    # image = F.normalize(image, dim=0)  # Normalize the image to have zero mean and unit variance
    return image.clone().requires_grad_(True)


def texture(model, gt: torch.Tensor):
    for param in model.parameters():
        param.requires_grad = False
    img_size = gt.size()
    cv2.imwrite("gt.jpg", gt.detach().numpy().transpose(1,2,0))
    tar = generate_white_noise(img_size)
    optimizer = torch.optim.Adam([tar], lr=0.001)
    model(gt)
    gt_grams = [feat.detach() for feat in extract_gram(model)]
    criterion = styleLoss(gt_grams)
    epoch = 50
    for i in tqdm(range(epoch)):
        # print(tar.grad)
        optimizer.zero_grad()
        model(tar)
        grams = extract_gram(model)
        loss = criterion(grams)
        loss.backward(retain_graph=True)
        optimizer.step()
        model.reset()
        print("epoch: {}, loss: {}".format(i, loss.item()))
        # print(tar.size())
        cv2.imwrite("epoch_{}.jpg".format(i), tar.detach().numpy().transpose(1,2,0))

    return tar


if __name__ == '__main__':
    model = get_vgg19_model()
    gt = cv2.imread("pebbles.jpg")
    gt = transforms.ToTensor()(gt).mul(255)
    # print(gt.size())
    tar = texture(model, gt)
    print(tar.size())
    pass
