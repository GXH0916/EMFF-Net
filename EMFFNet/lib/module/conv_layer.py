import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积模块，用于构建卷积操作，从输入数据中提取特征
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output

    # 批归一化 (Batch Normalization) 和 Parametric ReLU (PReLU) 激活函数的组合模块


# 用于深度神经网络中的卷积层或全连接层之后，提高模型的收敛速度和泛化能力
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)  # 帮助网络更好地适应数据分布和提高模型的表达能力

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = BNPReLU(32)
    print(model(x).shape)