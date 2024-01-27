"""
Author: Xiaohui Zhang, Siyuan Yin

calculate gram matrix
"""

import torch
import torch.nn as nn
from torchvision import transforms
from typing import List
import cv2
from VGG19 import get_vgg19_model
import torch
from tqdm import tqdm
import PIL.Image as Image
import torchvision.transforms.functional as TF
import numpy as np


def read_image(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485,0.456,0.406],
        #                      std=[0.229,0.224,0.225])
    ])
    image = cv2.imread(path)
    #转化成RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    save_image(image, "gt.jpg")
    image = image.unsqueeze(0)
    

    return image

def save_image(tensor, file_name):
    img_pil = TF.to_pil_image(tensor,mode='RGB')
    img_pil.save(file_name)

def get_gram(features_maps):
    # each element of feature maps is [batch_size, channel, height, width]
    for key in features_maps:
        # remove the batch_size dimension and convert it into [channel, height * width]
        features_maps[key] = features_maps[key].squeeze(0).view(features_maps[key].size(1), -1)
    
    gram_matrix = dict()
    # compute the gram matrix
    for key in features_maps:
        F = features_maps[key]
        N,M = F.shape
        G = torch.mm(F, F.t())/M
        gram_matrix[key] = G
        
    return gram_matrix


def gram_mse_loss(syn,gt):
    total_loss = 0
    for key in syn:
        N = syn[key].shape[0]
        loss = torch.sum((syn[key]-gt[key])**2)/N**2
        total_loss += loss
    return total_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def synthesis(model,gt, device=device):
    #生成一个与gt相同大小的随机噪声

    # for param in model.parameters():
    #     param.requires_grad = False
    model.to(device)
    gt = gt.to(device)

    syn = torch.rand(gt.shape)
    syn = syn.to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([syn], lr=0.5)
    # oprimizer2 = torch.optim.Adam(model.parameters(), lr=0.01)
    model(gt)
    gt_grams = get_gram(model.features_maps)

    epoch = 301
    for i in tqdm(range(epoch)):
        # print(tar.grad)
        optimizer.zero_grad()
        model(syn)
        syn_grams = get_gram(model.features_maps)
        loss = gram_mse_loss(syn_grams,gt_grams)
        # model.backward()
        loss.backward(retain_graph=True)

        optimizer.step()

        # 将syn的值限制在0-255之间
        syn.data = torch.clamp(syn.data,0,255)
        print("epoch: {}, loss: {}".format(i, loss.item()),flush=True)

        if i%10 == 0:
            save_image(syn.squeeze(0), "epoch_{}.jpg".format(i))

            


if __name__ == '__main__':
    model = get_vgg19_model(modified=True)
    gt = read_image("pebbles.jpg")
    synthesis(model, gt)
    # size = np.array((2,3))
    # M = np.prod(size)
    # print(M)

    

