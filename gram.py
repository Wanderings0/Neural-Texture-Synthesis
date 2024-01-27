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


def read_image(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485,0.456,0.406],
        #                      std=[0.229,0.224,0.225])
    ])
    image = cv2.imread(path)
    # print(max(image.flatten()), min(image.flatten()))
    #转化成RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)

    image = image.unsqueeze(0)

    return image

def save_image(tensor, file_name):
    img_pil = TF.to_pil_image(tensor)
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
        G = torch.mm(F, F.t())/(N*M*2)
        gram_matrix[key] = G
        
    return gram_matrix


def gram_mse_loss(syn,gt):
    total_loss = 0
    for key in syn:
        loss = torch.mean((syn[key]-gt[key])**2)
        total_loss += loss
    return total_loss

def synthesis(model,gt):
    #生成一个与gt相同大小的随机噪声

    for param in model.parameters():
        param.requires_grad = False

    syn = torch.rand(gt.shape)*255
    syn = syn.requires_grad_(True)
    optimizer = torch.optim.Adam([syn], lr=0.5)
    model(gt)
    gt_grams = get_gram(model.features_maps)

    epoch = 301
    for i in tqdm(range(epoch)):
        # print(tar.grad)
        optimizer.zero_grad()
        model(syn)
        syn_grams = get_gram(model.features_maps)
        loss = gram_mse_loss(syn_grams,gt_grams)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("epoch: {}, loss: {}".format(i, loss.item()),flush=True)

        if i%50 == 0:
            save_image(syn.squeeze(0), "epoch_{}.jpg".format(i))

            
            

    
#############################################


# def generate_white_noise(size):
#     image = torch.rand(size)
#     image = image * 255
#     cv2.imwrite("white_noise.jpg", image.detach().numpy().transpose(1,2,0))
#     # image = F.normalize(image, dim=0)  # Normalize the image to have zero mean and unit variance
#     # return image.clone().requires_grad_(True)
#     return image.clone()


# def texture(model, gt: torch.Tensor):
#     for param in model.parameters():
#         param.requires_grad = False
#     img_size = gt.squeeze(0).size()
#     # cv2.imwrite("gt.jpg", gt.detach().numpy().transpose(1,2,0))
#     tar = generate_white_noise(img_size).unsqueeze(0).requires_grad_(True)
#     optimizer = torch.optim.Adam([tar], lr=0.001)
#     model(gt)
#     gt_grams = [feat.detach() for feat in get_gram(model.features_maps).values()]
#     criterion = styleLoss(gt_grams)
#     epoch = 50
#     for i in tqdm(range(epoch)):
#         # print(tar.grad)
#         optimizer.zero_grad()
#         model(tar)
#         grams = [feat for feat in get_gram(model.features_maps).values()]
#         loss = criterion(grams)
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         print("epoch: {}, loss: {}".format(i, loss.item()))
#         # print(tar.size())
#         # cv2.imwrite("epoch_{}.jpg".format(i), tar.detach().numpy().transpose(1,2,0))

#     return tar


if __name__ == '__main__':
    model = get_vgg19_model()
    gt = read_image("pebbles.jpg")
    # texture(model, gt)
    synthesis(model, gt)
    

