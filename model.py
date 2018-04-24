# coding:utf-8
import torch
import torch.nn as nn
import numpy as np

# Pytorch两种定义网络的方式：残差网络和生成器模型的定义方式
# 含5层网络的残差块
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()   # super机制：子类（ResidualBlock）继承父类（nn.Module）。即允许子类使用父类的方法
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)     # ？


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        # Conv2d().__init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim     # num of filters
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers * 6.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            # '// '运算结果为整型
            # ConvTranspose2d 上采样
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # 最后一层用 Tanh 将输出图片的像素归一化至 -1~1，如果希望归一化至 0~1，则需使用 Sigmoid。
        layers.append(nn.Tanh())
        # nn库里有一个类型叫做Sequential序列，这个Sequential是一个容器类，我们可以在它里面添加一些基本的模块
        self.main = nn.Sequential(*layers)

    # x为输入图像，c为label向量
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

# 判别器没有normalization
# 生成器的激活函数用的是ReLU，而判别器使用的是LeakyReLU，二者并无本质区别，这里的选择更多是经验总结
# 每一个样本经过判别器后，输出一个 0~1 的数，表示这个样本是真图片的概率。
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        # 输入层
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        # 隐藏层
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))     # default: 128 / 64 = 2
        self.main = nn.Sequential(*layers)      # 为什么Sequential在这个位置？
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)            # h = [torch.FloatTensor of size 2x2048x2x2]，2大类，每类2048组数据，每组数据内容是一个2x2的矩阵，矩阵中的数值都非常小，eg，-1.1706e-05
        out_src = self.conv1(h)     # [torch.FloatTensor of size 2x1x2x2]
        out_cls = self.conv2(h)     # [torch.FloatTensor of size 2x5x1x1]
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))