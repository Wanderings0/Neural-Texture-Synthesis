import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import time, sys


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
                    x = inputs.cuda(0)
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
    

def main():
    # load the pretrained model
    vgg19_model = models.vgg19(weights='DEFAULT').cuda(0)

    for i, layer in enumerate(vgg19_model.features):
        if isinstance(layer, nn.MaxPool2d):
            vgg19_model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    print('VGG19 model with maxpooling replaced with average pooling:')
    print(vgg19_model)

    # 指定ImageNet数据集的路径
    imagenet_data_path = '/lustre/dataset/imagenet/'

    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载整个ImageNet数据集
    full_dataset = datasets.ImageFolder(root=imagenet_data_path, transform=transform)
    print(len(full_dataset))
    # 选择子集：这里我们随机选择一定比例的数据作为子集
    subset_size = len(full_dataset) // 3000  # 假设我们只取数据集的1/3000
    indices = torch.randperm(len(full_dataset)).tolist()
    subset_indices = indices[:subset_size]

    # 使用SubsetRandomSampler为我们的子集创建一个sampler
    sampler = SubsetRandomSampler(subset_indices)

    # 创建DataLoader来加载数据
    batch_size = 128
    dataloader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler)
    print(len(dataloader))
    # flush the stdout buffer
    sys.stdout.flush()

    rescale_weights(vgg19_model, dataloader)

    # save the model
    torch.save(vgg19_model.state_dict(), 'rescaled_vgg19_weights.pth')
    print('Model saved with path: rescaled_vgg19_weights.pth')

if __name__ == '__main__':
    # record the running time
    start_time = time.time()
    main()
    end_time = time.time()
    print('Running time: {:.2f} seconds.'.format(end_time - start_time))