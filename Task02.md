# 【Datawhale AI 夏令营】Deepfake攻防挑战赛 - 图像赛道_02
#AI夏令营#baseline#深度学习

## Part1 Deepfake赛题分析
### 1.1 对于deeffake图像的特征分析方法
  Deepfake是一种使用人工智能技术生成的伪造媒体，特别是视频和音频（我比较常见是图像、视频），它们看起来或听起来非常真实，但实际上是由计算机生成的。这种技术通常涉及到深度学习算法，特别是生成对抗网络（GANs），它们能够学习真实数据的特征，并生成新的、逼真的数据。
  *  对比正常图片和生成式图片：
        1. 人物细节：尤其是眼睛、嘴巴；
        2. 光影和阴影：分析光影是否一致以及阴影是否符合；
        3. 像素：放大图像查看像素是否存在模糊或像素化部分；
        4. 背景：背景中是否有不协调的元素，如主体边缘是否平滑或背景中不自然的重复。
  *  *插一下视频的对比：
        1. 帧间会出现突然的模糊-清晰的变换；
        2. 图像色彩会有更大模型或sd生成的；
        3. 单单基于无音源的视频很难作出区分，所以对声音进行识别区分就很重要（个人认为），
        比如音频中存在多次重复关键词，不自然的短句、结巴等。
###  1.2 deepfake检测思路
  1. 任务：对于给定的图像区分deepfake与否，可以将该任务视为图像二分类的问题。
  2. 注意事项：模型要求是单模型且参数量不得超过200M；
  3. 建模思路：构建一个基于CNN实现的二分类模型。
  4. 提分思路：
          -  对数据集进行数据增强，如旋转、颜色变换等；
          -  由于赛事要求模型为单模型，且模型的参数量也做了限制，则可以考虑更换在满足模型要求下最大的模型；
          -  对于训练时间上来说，可以提前对数据集进行预处理，如缩放。
## Part2 baseline步骤详解
  本次使用的模型是残差模型ResNet-18进行训练和推理，具体步骤如下（注：ResNet模型可以有效去除梯度消失带来的问题）：
  1. 数据加载和数据增强：
        数据增强，如随机裁剪、翻转、旋转等，以增加模型的泛化能力。
  2. 模型定义：
      1. 使用timm库中的预训练模型ResNet-18作为预训练模型；
      2. 调整模型最后的全连接层以适应当前任务的类别num_classes=2。
  3. 损失函数与优化器：
      1. 选择适当的损失函数，如交叉熵损失，用于衡量模型输出与真实标签之间的差异;
      2. 选择优化器（如 Adam），用于根据损失函数的梯度调整模型的权重。
  4. 训练过程：
      1. 循环执行epochs，遍历train_loader，计算损失、更新参数并计算准确率;
  5. 验证过程：
      1. 编写验证函数，用于评估模型在验证集上的性能。
      2. 在验证函数中，将模型设置为评估模式，遍历验证集数据，计算损失并评估预测准确率
  6. 性能评估：
      1. 使用准确率（Accuracy）作为主要性能评估指标，监控模型在每个 epoch 后在验证集上的表现。
      2. 最后，将模型用于生成预测结果，并将结果保存到 CSV 文件中，以便进一步分析或提交到竞赛平台
    
## Part3 重点代码
  数据增强：通过对数据集进行随机水平翻转、随机垂直翻转、随机旋转、随机灰度、随机裁剪等处理，以丰富数据集的多样性，提高模型的泛化能力。

       train_loader = torch.utils.data.DataLoader(
    FFDIDataset(train_label['path'], train_label['target'], 
                transforms.Compose([
                    transforms.Resize((312, 312)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.CenterCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    ), batch_size=40, shuffle=True, num_workers=4, pin_memory=True)
  
  模型优化：使用ResNet+SE注意力机制模块(Squeeze-and-Excitaion)，注意力机制可以增强模型提取特征的能力。SE模块通过对每个通道的特征进行加权，使得模型能够更好地捕捉到重要的特征，从而提升模型的拟合能力和整体性能。

    import torch
    import torch.nn as nn
    import torchvision.models.resnet as resnet
    
    import timm
    import torch
    import torch.nn as nn
    
    # 定义一个带有SE模块的Bottleneck块
    class SEBottleneck(nn.Module):
        expansion = 4  # 每个Bottleneck块的输出通道数是输入通道数的4倍
    
        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None, reduction=16):
            super(SEBottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # 定义基本的卷积层
            self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
            self.bn1 = norm_layer(width)
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   padding=1, groups=groups, bias=False, dilation=dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
    
            # 添加SE模块
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化，压缩空间维度
                nn.Conv2d(planes * self.expansion, planes * self.expansion // reduction, kernel_size=1),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(planes * self.expansion // reduction, planes * self.expansion, kernel_size=1),  # 升维
                nn.Sigmoid()  # 输出范围限制在0到1之间
            )
    
        def forward(self, x):
            identity = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
    
            out = self.conv3(out)
            out = self.bn3(out)
    
            if self.downsample is not None:
                identity = self.downsample(x)
    
            # SE模块
            se_out = self.se(out)
            out = out * se_out  # 将SE模块的输出与原特征图相乘
    
            out += identity
            out = self.relu(out)
    
            return out
    
    # 自定义ResNet50模型
    class ResNet50SE(nn.Module):
        def __init__(self, num_classes=1000, zero_init_residual=False,
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None):
            super(ResNet50SE, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer
    
            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                replace_stride_with_dilation = [False, False, False]
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(SEBottleneck, 64, 3)
            self.layer2 = self._make_layer(SEBottleneck, 128, 4, stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(SEBottleneck, 256, 6, stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(SEBottleneck, 512, 3, stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * SEBottleneck.expansion, num_classes)
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, SEBottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
    
        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion),
                )
    
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
    
            return nn.Sequential(*layers)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
    
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
    
            return x

    # 创建模型实例
    model = ResNet50SE(num_classes=2)
      
    # 打印模型结构
    print(model)
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

通过对数据集和模型的优化措施，在经过10个epoch后，模型在验证集上可以达到89.3%的准确率。表明通过数据增强和引入注意力机制，模型的特征提取能力和泛化能力得到了显著的提升。
