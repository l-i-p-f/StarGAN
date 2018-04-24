# coding:utf-8
from torch.utils import data
from torchvision import transforms as T     # PyTroch的数据预处理模块
from torchvision.datasets import ImageFolder
from PIL import Image   # torchvision.transform，只对 PIL.Image 或维度为 (H, W, C) 的图片数据进行数据预处理。由于 OpenCV 读入图片的数据维度是 (H, W, C)，所以不能直接使用 torchvision.transform 处理 OpenCV 的图片数据。
import torch
import os
import random

# 初始化及处理CelebA数据
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []     # 训练集
        self.test_dataset = []      # 测试集
        self.attr2idx = {}          # 属性转换成index，即用数字代表属性
                                    # attr2idx = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, ... 'Young': 39}
        self.idx2attr = {}          # 数字转换为属性。
                                    # idx2attr = {0: '5_o_Clock_Shadow', 1: 'Arched_Eyebrows', ... 39: 'Young'}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        # lines 存储attr_path中读取的内容，所有数据，类型均为str
        # lines[0]是数据大小，lines[1]是属性名称
        # 数据内容格式为[000001.jpg -1  1  ...  -1  1]，第一列为图片名称，后面为图片具有的属性，1表示有，-1表示没有
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)       # 打乱图片顺序

        # 根据所选属性制作label
        for i, line in enumerate(lines):
            split = line.split()    # 取一行数据line，把该行数据分开，存储到split
            filename = split[0]     # 首元素为图片名称
            values = split[1:]      # 后面元素为属性值，1或-1

            label = []
            # 判断当前图片是否具有所选属性（selected_attrs）
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')    # values记录了当前图片所有的属性

            # 选取2000张作为测试集
            if (i+1) <= 2:   # mark ori_=2000
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

# 下载完数据之后，将所有图片放在一个文件夹，然后将该文件夹移动至 data 目录下（请确保data下没有其他的文件夹）。
# 这种处理方式是为了能够直接使用torchvision自带的ImageFolder读取图片，而不必自己写Dataset。
def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())  # 随机水平翻转，0.5概率翻转。
    transform.append(T.CenterCrop(crop_size))       # 中心切割，这里将图片切割成178 x 178的图像
    transform.append(T.Resize(image_size))          # resize图像大小为128 x 128
    # ToTensor：把取值范围[0,255]的image或者shape为(H,W,C)的numpy.ndarray，转换为(C,H,W)，取值范围[0,1]的torch.FloatTensor
    transform.append(T.ToTensor())
    # Normalize：给定均值mean=(R,G,B) 方差std=(R,G,B)，将Tensor归一化。即：Normalized_image = (image - mean)/std。
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)                # 输入一个transform列表，将多个transform组合使用。

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        # ImageFolder是PyTorch中一个通用的数据加载器，数据集中的数据以以下方式组织
        # image_dir / dog / xxy.png
        # image_dir / dog / xxz.png
        # ...
        # image_dir / cat / 123.png
        # 他有以下成员变量:
        # self.classes - 用一个list保存类名
        # self.class_to_idx - 类名对应的索引
        # self.imgs - 保存(img - path,class ) tuple的list
        dataset = ImageFolder(image_dir, transform)

    # 加载数据，DataLoader提供了队列和线程
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)  # num_workers=k表示使用k个子进程来加载数据
    return data_loader