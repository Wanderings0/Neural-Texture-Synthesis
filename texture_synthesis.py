"""
Author: Xiaohui Zhang, Siyuan Yin

Synthesize an image from a given image using VGG19 model
"""

import torch
from torchvision import transforms
import cv2
from VGG19 import get_vgg19_model,rescale_weights
import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np
import argparse


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
    # save_image(image, "gt.jpg")
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


def gram_mse_loss(syn,gt,feature_selection=['conv1_1','conv2_1','conv3_1','conv4_1']):
    total_loss = 0
    for key in syn:
        if key not in feature_selection:
            continue
        N = syn[key].shape[0]
        loss = torch.sum((syn[key]-gt[key])**2)/N**2
        total_loss += loss
    return total_loss



def synthesis(model,gt, args):
    # for param in model.parameters():
    #     param.requires_grad = False
    device = args.device
    model.to(device)
    gt = gt.to(device)

    syn = torch.rand(gt.shape)
    syn = syn.to(device).requires_grad_(True)

    model(gt)
    gt_grams = get_gram(model.features_maps)

    epoch = args.epoch

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam([syn], lr=args.lr)
        for i in tqdm(range(epoch)):
        # print(tar.grad)
            optimizer.zero_grad()
            model(syn)
            syn_grams = get_gram(model.features_maps)
            loss = gram_mse_loss(syn_grams,gt_grams,args.feature_selection)
            # model.backward()
            loss.backward(retain_graph=True)

            optimizer.step()

            # 将syn的值限制在0-255之间
            syn.data = torch.clamp(syn.data,0,1)
            # print("epoch: {}, loss: {}".format(i, loss.item()),flush=True)

            if (i+1)%100 == 0:
                save_image(syn.squeeze(0), "epoch_{}.jpg".format(i+1))
    elif args.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS([syn], lr=args.lr)

        def closure():
            optimizer.zero_grad()
            model(syn)
            syn_grams = get_gram(model.features_maps)
            loss = gram_mse_loss(syn_grams,gt_grams,args.feature_selection)
            # model.backward()
            loss.backward(retain_graph=True)

            return loss
        
        for i in tqdm(range(epoch)):
            # print(tar.grad)

            optimizer.step(closure)

            # 将syn的值限制在0-255之间
            syn.data = torch.clamp(syn.data,0,1)
            # print("epoch: {}, loss: {}".format(i, loss.item()),flush=True)

            if (i+1)%100 == 0:
                save_image(syn.squeeze(0), "epoch_{}.jpg".format(i+1))
    else:
        raise NotImplementedError
    
    

    save_image(syn.squeeze(0), args.save_path)




def main():

    parser = argparse.ArgumentParser(description='PyTorch VGG19 Training')
    parser.add_argument('--model', default='vgg19', type=str, help='model name')
    parser.add_argument('--gt_path', default='leaf.jpg', type=str, help='path to ground truth image')
    parser.add_argument('--pool', default='avg', type=str, help='pooling method')
    parser.add_argument('--rescale', default=True, type=bool, help='rescale weights or not')
    parser.add_argument('--feature_selection', default=['conv1_1','conv2_1','conv3_1','conv4_1'], type=list, help='feature selection')
    
    
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimize method')
    parser.add_argument('--epoch', default=1000, type=int, help='epoch')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--save_path', default='result.jpg', type=str, help='save path') 
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    path = 'imgs/'+args.gt_path
    model = get_vgg19_model(pool=args.pool)
    gt = read_image(path)
    if args.rescale:
        rescale_weights(model,[gt],device)

    synthesis(model, gt, args)
    


if __name__ == '__main__':
    main()


