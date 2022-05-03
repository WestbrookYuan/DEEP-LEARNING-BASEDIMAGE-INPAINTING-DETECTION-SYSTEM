import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from Conv2dSame import conv2d_same_padding

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, filter_type="d1"):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.weights = []
        FILTERS = {
            'd1': [
                np.array([[0., 0., 0.],
                          [0., -1., 0.],
                          [0., 1., 0.]]),
                np.array([[0., 0., 0.],
                          [0., -1., 1.],
                          [0., 0., 0.]]),
                np.array([[0., 0., 0.],
                          [0., -1., 0.],
                          [0., 0., 1.]])],
            'd2': [
                np.array([[0., 1., 0.],
                          [0., -2., 0.],
                          [0., 1., 0.]]),
                np.array([[0., 0., 0.],
                          [1., -2., 1.],
                          [0., 0., 0.]]),
                np.array([[1., 0., 0.],
                          [0., -2., 0.],
                          [0., 0., 1.]])],
            'd3': [
                np.array([[0., 0., 0., 0., 0.],
                          [0., 0., -1., 0., 0.],
                          [0., 0., 3., 0., 0.],
                          [0., 0., -3., 0., 0.],
                          [0., 0., 1., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., -1., 3., -3., 1.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0.],
                          [0., -1., 0., 0., 0.],
                          [0., 0., 3., 0., 0.],
                          [0., 0., 0., -3., 0.],
                          [0., 0., 0., 0., 1.]])],
            'd4': [
                np.array([[0., 0., 1., 0., 0.],
                          [0., 0., -4., 0., 0.],
                          [0., 0., 6., 0., 0.],
                          [0., 0., -4., 0., 0.],
                          [0., 0., 1., 0., 0.]]),
                np.array([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [1., -4., 6., -4., 1.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]),
                np.array([[1., 0., 0., 0., 0.],
                          [0., -4., 0., 0., 0.],
                          [0., 0., 6., 0., 0.],
                          [0., 0., 0., -4., 0.],
                          [0., 0., 0., 0., 1.]])],
        }

        for kernel in FILTERS[filter_type]:
            kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
            kernel = np.repeat(kernel, self.channels, axis=0)
            self.weight = nn.Parameter(torch.Tensor(kernel), requires_grad=True).cuda()
            self.weights.append(self.weight)

    def __call__(self, x):
        x1 = conv2d_same_padding(x, self.weights[0], padding=2, groups=self.channels)
        x2 = conv2d_same_padding(x, self.weights[1], padding=2, groups=self.channels)
        x3 = conv2d_same_padding(x, self.weights[2], padding=2, groups=self.channels)
        return torch.cat([x1, x2, x3], 1)

