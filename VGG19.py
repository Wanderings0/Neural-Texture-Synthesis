''' 
Author: Siyuan Yin
import this file and call get_vgg19_model() to get the model
the model weight is already rescaled and should be saved in 'rescaled_vgg19_weights.pth'
get the feature maps by calling model.get_feature_maps()

example:
    # modified=True means the model is rescaled, modified=False means the model is original.
    model = get_vgg19_model(modified=True)

    feature_maps = model.features_maps

    gram_matrix = model.get_gram()
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import time, sys


# modify the path to your own path
# you can ignore the imagenet_path because the model is already rescaled and saved
model_path = 'rescaled_vgg19_weights.pth'
imagenet_path = '/lustre/dataset/imagenet/'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGG19(nn.Module):
    '''
    VGG19 model with maxpooling replaced with average pooling and without 3 FC layers, 16 conv layers in total
    '''
    def __init__(self):
        super(VGG19, self).__init__()

        # the shape of each element of feature maps is [batch_size, channel, height, width] 
        self.features_maps = dict()
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_4 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        # reset feature maps at the beginning of forward pass
        self.features_maps = dict()
        temp = x.clone()
        x = self.conv1_1(temp)
        x = self.relu1_1(x)
        self.features_maps["conv1_1"] = x
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        self.features_maps["conv1_2"] = x
        x = self.maxpool1(x)
        self.features_maps["pool1"] = x
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        self.features_maps["conv2_1"] = x
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        self.features_maps["conv2_2"] = x
        x = self.maxpool2(x)
        self.features_maps["pool2"] = x
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        self.features_maps["conv3_1"] = x
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        self.features_maps["conv3_2"] = x
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        self.features_maps["conv3_3"] = x
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        self.features_maps["conv3_4"] = x
        x = self.maxpool3(x)
        self.features_maps["pool3"] = x
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        self.features_maps["conv4_1"] = x
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        self.features_maps["conv4_2"] = x
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        self.features_maps["conv4_3"] = x
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        self.features_maps["conv4_4"] = x
        x = self.maxpool4(x)
        self.features_maps["pool4"] = x
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        self.features_maps["conv5_1"] = x
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        self.features_maps["conv5_2"] = x
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        self.features_maps["conv5_3"] = x
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        self.features_maps["conv5_4"] = x
        x = self.maxpool5(x)
        self.features_maps["pool5"] = x
        
        return x
    


class modifiedVGG19(nn.Module):
    '''
    VGG19 model with maxpooling replaced with average pooling and without 3 FC layers, 16 conv layers in total
    '''
    def __init__(self):
        super(modifiedVGG19, self).__init__()

        self.features_maps = dict()

        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_4 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_4 = nn.ReLU()
        self.avgpool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_4 = nn.ReLU()
        self.avgpool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        # reset feature maps at the beginning of forward pass
        self.features_maps = dict()
        temp = x.clone()
        x = self.conv1_1(temp)
        x = self.relu1_1(x)
        self.features_maps["conv1_1"] = x
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        self.features_maps["conv1_2"] = x
        x = self.avgpool1(x)
        self.features_maps["pool1"] = x
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        self.features_maps["conv2_1"] = x
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        self.features_maps["conv2_2"] = x
        x = self.avgpool2(x)
        self.features_maps["pool2"] = x
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        self.features_maps["conv3_1"] = x
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        self.features_maps["conv3_2"] = x
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        self.features_maps["conv3_3"] = x
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        self.features_maps["conv3_4"] = x
        x = self.avgpool3(x)
        self.features_maps["pool3"] = x
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        self.features_maps["conv4_1"] = x
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        self.features_maps["conv4_2"] = x
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        self.features_maps["conv4_3"] = x
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        self.features_maps["conv4_4"] = x
        x = self.avgpool4(x)
        self.features_maps["pool4"] = x
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        self.features_maps["conv5_1"] = x
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        self.features_maps["conv5_2"] = x
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        self.features_maps["conv5_3"] = x
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        self.features_maps["conv5_4"] = x
        x = self.avgpool5(x)
        self.features_maps["pool5"] = x
        
        return x


        


def rescale_weights(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                # 初始化累加器
                activation_sum = 0
                num_activations = 0
                
                print('Rescaling weights for layer {}...'.format(i))
                sys.stdout.flush()
                for inputs, _ in tqdm(dataloader):
                    # 仅计算当前层的前向传播结果
                    x = inputs.to(device)
                    for j in range(i + 1):
                        x = model.features[j](x)
                    
                    # 累加激活值的总和
                    activation_sum += x.sum(dim=[0, 2, 3])
                    num_activations += x.size(0) * x.size(2) * x.size(3)

                # 计算平均激活值
                mean_activation = activation_sum / num_activations

                # 调整权重使得平均激活值为1
                scale_factor = mean_activation
                layer.weight.data /= scale_factor[:, None, None, None]

                # 如果下一层是ReLU或者是最后一个卷积层，则跳过
                if i+1 < len(model.features) and isinstance(model.features[i+1], nn.Conv2d):
                    next_layer = model.features[i+1]
                    # 对下一层权重进行反向缩放
                    next_layer.weight.data *= scale_factor[None, :, None, None]

    
def get_vgg19_model(modified=True):

    if modified:
        # load a VGG19 model with rescaled weights ,replace the maxpooling with average pooling and without FC layers
        vgg19_model = modifiedVGG19()
        keys = list(vgg19_model.state_dict().keys())
        # print(keys)
        rescaled_weights = torch.load(model_path,map_location=device)
        for i, key in enumerate(rescaled_weights):
            if i < len(keys):
                # print(keys[i],"with weight ", key)
                vgg19_model.state_dict()[keys[i]] = rescaled_weights[key]
            else:
                break
    else:
        # load a VGG19 model with pretrained weights without FC layers
        vgg19_model = VGG19()
        pretrained_weights = models.vgg19(pretrained=True).state_dict()
        keys = list(vgg19_model.state_dict().keys())
        for i, key in enumerate(pretrained_weights):
            if i < len(keys):
                vgg19_model.state_dict()[keys[i]] = pretrained_weights[key]
            else:
                break
        
    return vgg19_model


def main():
    '''
    rescale the weights of VGG19 model and save the model
    '''
    # load the pretrained model
    vgg19_model = models.vgg19(weights='DEFAULT').to(device)

    for i, layer in enumerate(vgg19_model.features):
        if isinstance(layer, nn.MaxPool2d):
            vgg19_model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    print('VGG19 model with maxpooling replaced with average pooling:')
    print(vgg19_model)

    # 指定ImageNet数据集的路径
    imagenet_data_path = imagenet_path

    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载整个ImageNet数据集
    full_dataset = datasets.ImageFolder(root=imagenet_data_path, transform=transform)
    # print(len(full_dataset))
    # 选择子集：这里我们随机选择一定比例的数据作为子集
    subset_size = len(full_dataset) // 3000  # 假设我们只取数据集的1/3000
    indices = torch.randperm(len(full_dataset)).tolist()
    subset_indices = indices[:subset_size]

    # 使用SubsetRandomSampler为我们的子集创建一个sampler
    sampler = SubsetRandomSampler(subset_indices)

    # 创建DataLoader来加载数据
    batch_size = 128
    dataloader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler)
    # print(len(dataloader))
    # flush the stdout buffer
    # sys.stdout.flush()

    rescale_weights(vgg19_model, dataloader)

    # save the model
    torch.save(vgg19_model.state_dict(), model_path)
    print(f'Model saved with path: {model_path}')


if __name__ == '__main__':
    # record the running time
    # start_time = time.time()
    # main()
    # end_time = time.time()
    # print('Running time: {:.2f} seconds.'.format(end_time - start_time))
    model = get_vgg19_model(modified=True)
    # 随机生成图片并输入模型
    x = torch.rand(1, 3, 256, 256).to(device)
    y = model(x)
    # 获取特征图
    feature_maps = model.features_maps
    print(feature_maps['conv1_1'].shape)