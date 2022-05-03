import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# 对所有图片生成path-label map.txt 这个程序可根据实际需要适当修改


class MyDatasets(Dataset):

    def __init__(self, dir):
        # 获取数据存放的dir
        # 例如d:/images/
        self.data_dir = dir
        # 用于存放(image,label) tuple的list,存放的数据例如(d:/image/1.png,4)
        self.image_target_list = []
        self.label_target_list = []
        # 从dir--label的map文件中将所有的tuple对读取到image_target_list中
        # map.txt中全部存放的是d:/.../image_data/1/3.jpg 1 路径最好是绝对路径
        with open(os.path.join(dir, 'imagesmap.txt'), 'r') as fp:
            content = fp.readlines()
            # s.rstrip()删除字符串末尾指定字符（默认是字符）
            # 得到 [['d:/.../image_data/1/3.jpg', '1'], ...,]
            str_list = [s.rstrip().split() for s in content]
            # 将所有图片的dir--label对都放入列表，如果要执行多个epoch，可以在这里多复制几遍，然后统一shuffle比较好
            self.image_target_list = [x for x in str_list]
        self.label_target_list = np.load(dir + '/labels.npy')

    def __getitem__(self, index):
        image = self.image_target_list[index]
        # 按path读取图片数据，并转换为图片格式例如[3,32,32]
        # 可以用别的代替
        img = cv2.imread(image[0])
        input_x = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32))).permute(2, 0, 1)
        return input_x, self.label_target_list[index]

    def __len__(self):
        return len(self.image_target_list)