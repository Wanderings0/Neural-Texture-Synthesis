import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler



def rescale_weights(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                # 初始化累加器
                activation_sum = 0
                num_activations = 0

                for inputs, _ in dataloader:
                    # 仅计算当前层的前向传播结果
                    x = inputs.cuda()
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
    vgg19_model = models.vgg19(pretrained=True).cuda()

    for i, layer in enumerate(vgg19_model.features):
        if isinstance(layer, nn.MaxPool2d):
            vgg19_model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    # 指定ImageNet数据集的路径
    imagenet_data_path = '/path/to/imagenet'

    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载整个ImageNet数据集
    full_dataset = datasets.ImageFolder(root=imagenet_data_path, transform=transform)

    # 选择子集：这里我们随机选择一定比例的数据作为子集
    subset_size = len(full_dataset) // 10  # 假设我们只取数据集的1/10
    indices = torch.randperm(len(full_dataset)).tolist()
    subset_indices = indices[:subset_size]

    # 使用SubsetRandomSampler为我们的子集创建一个sampler
    sampler = SubsetRandomSampler(subset_indices)

    # 创建DataLoader来加载数据
    batch_size = 32
    dataloader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler)

    rescale_weights(vgg19_model, dataloader)

    # save the model
    torch.save(vgg19_model.state_dict(), 'rescaled_vgg19_weights.pth')