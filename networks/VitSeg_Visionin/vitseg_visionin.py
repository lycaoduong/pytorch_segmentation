import torch
from torch import nn
from networks.swin_transformer.swin_transformer import get_Swin_TR
import torch.nn.functional as F
from networks.efficientnet.utils import MemoryEfficientSwish, Swish
from networks.efficientnet.utils_extra import Conv2dStaticSamePadding
from networks.efficientdet.model import RetinaNet, BiFPN
from networks.effieicentFormer import efficientformer


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class Segment_Head(nn.Module):
    """
    customized by LCD
    """
    def __init__(self, img_size, in_ch, out_ch, num_layers, pyramid_levels, onnx_export=False):
        super(Segment_Head, self).__init__()
        self.img_size = img_size
        self.outconv = nn.Conv2d(in_ch*4, out_ch, 1)
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_ch, in_ch, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.num_layers = num_layers
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            # feat = self.header(feat)
            # feat = feat.permute(0, 2, 3, 1)
            feat = F.interpolate(feat, size=torch.Size([self.img_size, self.img_size]), mode="bilinear",
                                 align_corners=True)
            feats.append(feat)
        out = self.outconv(torch.cat(feats, 1))
        return out


class VitSegVisionin(nn.Module):
    def __init__(self, backbone, img_size=512, in_ch=3, out_ch=1, onnx_export=True, **kwargs):
        super(VitSegVisionin, self).__init__()
        if backbone == 'Swin-T':
            self.backbone_net = get_Swin_TR(backbone, img_size=img_size, in_ch=in_ch)
        if backbone == 'Swin-S':
            self.backbone_net = get_Swin_TR(backbone, img_size=img_size, in_ch=in_ch)
        if backbone == 'Swin-B':
            self.backbone_net = get_Swin_TR(backbone, img_size=img_size, in_ch=in_ch)
        if backbone == 'Swin-L':
            self.backbone_net = get_Swin_TR(backbone, img_size=img_size, in_ch=in_ch)
        if backbone == 'effFomer-L1':
            self.backbone_net = efficientformer.efficientformer_l1(fork_feat=True)
        if backbone == 'effFomer-L3':
            self.backbone_net = efficientformer.efficientformer_l3(fork_feat=True)
        if backbone == 'effFomer-L7':
            self.backbone_net = efficientformer.efficientformer_l7(fork_feat=True)

        self.fpn_filter = 64
        self.fpn_repeat = 3
        self.pyramid_level = 4

        conv_channel_coef = {
            # the channels of P3/P4/P5.
            'Swin-T':                [192, 384, 768],
            'Swin-S':                [192, 384, 768],
            'Swin-B':                [256, 512, 1024],
            'Swin-L':                [384, 768, 1536],
            'effFomer-L1':           [96, 224, 448],
            'effFomer-L3':           [128, 320, 512],
            'effFomer-L7':           [192, 384, 768],
            'resnet-34':             [128, 256, 512],
            'resnet-50':             [512, 1024, 2048],
            'resnet-101':            [512, 1024, 2048],
        }

        self.neck = nn.Sequential(
            *[BiFPN(self.fpn_filter,
                    conv_channel_coef[backbone],  # default compound_coef
                    True if _ == 0 else False,
                    attention=True,
                    onnx_export=onnx_export,
                    use_p8=False)
              for _ in range(self.fpn_repeat)])

        # self.neck = RetinaNet(C3_size=conv_channel_coef[backbone][0],
        #                       C4_size=conv_channel_coef[backbone][1],
        #                       C5_size=conv_channel_coef[backbone][2],
        #                       feature_size=self.fpn_filter)

        self.segmentation = Segment_Head(img_size=img_size, in_ch=self.fpn_filter, out_ch=out_ch, num_layers=self.fpn_repeat, pyramid_levels=self.pyramid_level, onnx_export=onnx_export)

    def forward(self, inputs):
        o = self.backbone_net(inputs)
        fpn_feature = (o[-3], o[-2], o[-1])
        fpn_feature = self.neck(fpn_feature)
        d0 = self.segmentation(fpn_feature)
        # Ignore if Loss function has activation
        prediction = torch.sigmoid(d0)
        return prediction


if __name__ == '__main__':
    device = 'cuda'
    backbone = 'Swin-L'
    img_size = 512
    data = torch.randn((1, 3, img_size, img_size)).to(device)
    model = VitSegVisionin(backbone=backbone, img_size=img_size, in_ch=3, out_ch=6).to(device)
    pred = model(data)
    print(pred)
    print(pred.shape)

    # import timm
    # import torch
    #
    # model = timm.create_model('resnet34', pretrained=False, features_only=True)
    # x = torch.randn(1, 3, 512, 512)
    # o = model(x)
    # print(1)