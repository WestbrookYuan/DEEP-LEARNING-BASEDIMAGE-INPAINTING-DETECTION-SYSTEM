import torch

from Graduation_Pytorch import Residual, ResNet_v2
from bilinear_upsample_weights import bilinear_upsample_weights as upsample
from torch.nn import functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gc = Residual.GaussianBlurConv()
        self.resnet = ResNet_v2.PreActResNet(ResNet_v2.PreActBottleneck, [2, 2, 2, 2])
        self.upweights1 = torch.nn.Parameter(torch.Tensor(upsample(4, 64, 1024)).permute(3, 2, 1, 0),
                                             requires_grad=True).cuda()
        self.upweights2 = torch.nn.Parameter(torch.Tensor(upsample(4, 4, 64)).permute(3, 2, 1, 0),
                                             requires_grad=True).cuda()
        self.BN = torch.nn.BatchNorm2d(4)
        self.ReLU = torch.nn.ReLU()
        self.Conv2d = torch.nn.Conv2d(4, 2, (5, 5))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):
        data = self.gc(data)
        data = self.resnet(data)
        data = F.conv_transpose2d(data, self.upweights1, stride=(4, 4), padding=(1, 1))
        data = F.conv_transpose2d(data, self.upweights2, stride=(4, 4), padding=(4, 4))
        data = self.BN(data)
        data = self.ReLU(data)
        data = self.Conv2d(data)
        predicted = self.softmax(data)
        zeros = torch.zeros_like(predicted)
        ones = torch.ones_like(predicted)
        predicted_mask = torch.where(predicted >= 0.5, ones, zeros)
        return predicted, predicted_mask
