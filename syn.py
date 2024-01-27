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


def gram_mse_loss(activations, target_gram_matrix, weight=1., linear_transform=None):
    '''
    This function computes an elementwise mean squared distance between the gram matrices of the source and the generated image.

    :param activations: the network activations in response to the image that is generated
    :param target_gram_matrix: gram matrix in response to the source image
    :param weight: scaling factor for the loss function
    :param linear_transform: linear transform that is applied to the feature vector at all positions before gram matrix computation
    :return: mean squared distance between normalised gram matrices and gradient wrt activations
    '''

    N = activations.shape[1]
    fm_size = np.array(activations.shape[2:])
    M = np.prod(fm_size)
    G_target = target_gram_matrix
    if linear_transform == None:
        F = activations.reshape(N,-1) 
        G = np.dot(F,F.T) / M
        loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
        gradient = (weight * np.dot(F.T, (G - G_target)).T / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
    else: 
        F = np.dot(linear_transform, activations.reshape(N,-1))
        G = np.dot(F,F.T) / M
        loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
        gradient = (weight * np.dot(linear_transform.T, np.dot(F.T, (G - G_target)).T) / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
        
    return [loss, gradient]

def ImageSyn(model,gram_matrix,gt):
    init = torch.randn(gt.shape)*255

    layers = model.features_maps.keys()

    def f(x):
        x = x.reshape(gt.shape)
        model(x)
        f_val=0
        # clear gradient in all layers
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        for i,index in enumerate(layers):
            layer = layers[index]
            



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


if __name__ == "__main__":
    model = get_vgg19_model()
    gt = read_image('pebbles.jpg')
    gt = model(gt)
    gram_matrix = get_gram(model.features_maps)
    ImageSyn(model,gram_matrix,gt)

    