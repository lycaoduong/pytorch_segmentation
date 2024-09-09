from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torchvision.models import resnet
import torch
from torch import nn

class DeepLabV3_ly(nn.Module):
    """
    customized by LCD
    """
    def __init__(self, in_ch, out_ch, backbone=None, pretrain=False):
        super(DeepLabV3_ly, self).__init__()
        if backbone == 'resnet101':
            self.model = deeplabv3_resnet101(num_classes=out_ch, pretrained=pretrain)
        else:
            self.model = deeplabv3_resnet50(num_classes=out_ch, pretrained=pretrain)
        self.model.backbone.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    def forward(self, inputs):
        prediction = self.model(inputs)['out']
        prediction = torch.sigmoid(prediction)
        return prediction

if __name__ == '__main__':
    device = 'cuda'
    data = torch.randn((2, 2, 512, 512)).to(device)
    model = DeepLabV3_ly(in_ch=2, out_ch=6, pretrain=False).to(device)
    for i in range(100):
        pred = model(data)
        print(i)
        # print(pred.shape)