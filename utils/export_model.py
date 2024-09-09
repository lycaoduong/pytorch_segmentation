import cv2
from networks.u2net.u2net import U2NET, U2NETP
from networks.VitSeg_Visionin.vitseg_visionin import VitSegVisionin
from networks.deeplabv3.deeplabv3 import DeepLabV3_ly
from networks.unet.unet_model import UNet
import os
import torch


def export_onnx_model(model, weighted, save_path, model_name, device='cpu', img_size=256, channel=3):
    os.makedirs(save_path, exist_ok=True)
    full_model_path = os.path.join(save_path, model_name)

    model = model.to(device)
    weight = torch.load(weighted, map_location=device)
    model.load_state_dict(weight, strict=True)
    model.eval()
    generated_input = torch.randn((1, channel, img_size, img_size)).to(device)
    torch.onnx.export(
        model,
        generated_input,
        full_model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None
    )

if __name__ == '__main__':
    # channel = 2
    # save_path = 'C:/Users/user/Documents/GitHub/runs/train/HyundaiSAM/unet_hyundai_sam_v2/single_run/2023.05.14_12.58.01'
    # model_name = 'hyundai_pro_last.onnx'
    # model_ckpt = 'last.pt'

    # model = VitSegVisionin(backbone="Swin-L", img_size=512, in_ch=channel, out_ch=6, onnx_export=True)
    # model = UNet(n_channels=2, n_classes=6, bilinear=False)
    # model = U2NET(in_ch=channel, out_ch=6)
    # model = DeepLabV3_ly(in_ch=channel, out_ch=6, pretrain=False)

    channel = 3
    out = 2
    model = U2NET(in_ch=channel, out_ch=out)
    save_path = '../models/'
    model_name = 'visionin_wood.onnx'
    model_ckpt = 'u2net_human_seg.pth'

    weighted = '{}/{}'.format(save_path, model_ckpt)
    export_onnx_model(model, weighted=weighted, save_path=save_path, model_name=model_name, device='cpu', img_size=320, channel=channel)
