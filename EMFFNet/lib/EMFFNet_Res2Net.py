import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.module.SCEConv import SCEConv
from lib.module.SFF import SqueezeAndExciteFusionAdd
from lib.module.cbam import CBAMBlock
from lib.module.gct import GCT
from lib.module.SCSA import SCSA
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# RFB模块
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

# partial decoder
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)
    def forward(self, x1, x2, x3, x4):
        x1_1 = x1   # channels = 32
        # x4_rfb (1,32,11,11)  x1_1(1,32,11,11)
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        # channels = 32
        # x2_1(1,32,22,22)
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        # x3_1 (1,32,44,44)
        x4_1 = self.conv_upsample4(self.upsample(self.upsample(self.upsample(x1))))\
               * self.conv_upsample4(self.upsample(self.upsample(x2)))\
               * self.conv_upsample4(self.upsample(x3)) * x4
        # x4_1(1,32,88,88)

        # channels = 32
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        #x2_2(1,64,22,22)

        # channels = 64
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        # x3_2(1,96,44,44)

        x4_2 = torch.cat((x4_1, self.conv_upsample6(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)
        # x4_2(1,128,88,88)

        # channels = 96
        x = self.conv4(x4_2)    # channels = 96
        x = self.conv5(x)       # channels = 1

        return x



class EMFFNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, selfnorm=None):
        super(EMFFNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb1_1 = RFB_modified(256, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- add module
        self.SCEConv_1 = SCEConv(2048, 2048)
        self.SCEConv_2 = SCEConv(1024, 1024)
        self.SCEConv_3 = SCEConv(512, 512)
        self.SCEConv_4 = SCEConv(256, 256)
        self.cbam = CBAMBlock(32)
        self.SCSA = SCSA(32,32)
        self.gct = GCT(1)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        self.sff = SqueezeAndExciteFusionAdd(256)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 1 ----
        self.ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # sff1
        self.upsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(32, 1, kernel_size=1)

        # x1
        self.conv_x1 = BasicConv2d(64, 256, kernel_size=1)
        self.upsample_x1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        x = self.conv_x1(x)
        x = self.SCEConv_4(x)
        x1 = self.SCEConv_4(x1)
        sff1 = self.sff(x, x1)


        x1_rfb = self.rfb1_1(sff1)    # channel -> 32   1, 32, 88, 88
        x2_rfb = self.rfb2_1(x2)      # channel -> 32   1, 32, 44, 44
        x3_rfb = self.rfb3_1(x3)      # channel -> 32   1, 32, 22, 22
        x4_rfb = self.rfb4_1(x4)      # channel -> 32   1, 32, 11, 11

        x1_rfb = self.SCSA(x1_rfb)
        x2_rfb = self.SCSA(x2_rfb)
        x3_rfb = self.SCSA(x3_rfb)
        x4_rfb = self.SCSA(x4_rfb)


        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb, x1_rfb) #1, 1, 88, 88

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=4, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.125, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = self.gct(x)
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.SCEConv_1(x)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat_1 = self.ra4_conv5(x)
        x = ra4_feat_1 + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = self.gct(x)
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.SCEConv_2(x)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat_1 = self.ra3_conv4(x)
        x = ra3_feat_1 + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = self.gct(x)
        x = x.expand(-1, 512, -1, -1).mul(x2)
        # x(1, 1, 44, 44) x2 (1, 512, 44, 44)
        x = self.SCEConv_3(x)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat_1 = self.ra2_conv4(x)
        x = ra2_feat_1 + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_1 ----
        # 1, 1, 44, 44
        crop_1 = F.interpolate(x, scale_factor=2, mode='bilinear') #  1, 1, 352, 352
        x = -1 * (torch.sigmoid(crop_1)) + 1
        x = self.gct(x)
        x = x.expand(-1, 256, -1, -1).mul(x1)
        # x (1, 1, 88, 88) x1(1, 256, 88, 88)
        x = self.SCEConv_4(x)
        x = self.ra1_conv1(x)
        x = F.relu(self.ra1_conv2(x))
        x = F.relu(self.ra1_conv3(x))
        ra1_feat_1 = self.ra1_conv4(x)
        x = ra1_feat_1 + crop_1
        lateral_map_1 = F.interpolate(x, scale_factor=4,mode='bilinear')  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)


        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1


if __name__ == '__main__':
    model = EMFFNet(32)


    ras = EMFFNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
    # print(model)