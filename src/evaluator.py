import datetime
import os
import json
from utils.utils import YamlRead
from utils import transform as tr
from torchvision import transforms
from utils.data_loader import load_visionin_wood
from torch.utils.data import DataLoader
from networks.u2net.u2net import U2NET, U2NETP
from networks.VitSeg_Visionin.vitseg_visionin import VitSegVisionin
from networks.deeplabv3.deeplabv3 import DeepLabV3_ly
from networks.segmenter.segmenter import Segmenter_Ly
from networks.unet.unet_model import UNet
from torch import nn
from utils.loss_function import Custom_Loss, confusion_matrix_ly
import torch
from tqdm.autonotebook import tqdm
import traceback
from tensorboardX import SummaryWriter
import numpy as np


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_func_type='dice', reduction='mean', class_w=None):
        super().__init__()
        self.model = model
        self.criterion = Custom_Loss(loss_type=loss_func_type, reduction=reduction)
        self.weight = class_w

    def forward(self, image, mask, **kwargs):
        output = self.model(image)
        losses = self.criterion(output, mask, self.weight)
        cm_cls = confusion_matrix_ly(output, mask)
        return losses, cm_cls

    def run(self, image, id_token, mask_token, mask, **kwargs):
        output = self.model(image, id_token, mask_token)
        losses = self.criterion(output, mask, self.weight)
        cm_cls = confusion_matrix_ly(output, mask)
        return losses, cm_cls


class Eval(object):
    def __init__(self, eval_opt):
        self.project = eval_opt.project
        self.model_name = eval_opt.model
        self.dataset = eval_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)


        exp_name, trial_name = 'single_run', date_time

        self.save_dir = '../runs/eval/{}/{}_{}/{}/{}/'.format(self.project, self.model_name, self.dataset, exp_name, trial_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.logs = self.save_dir + 'logs/'
        os.makedirs(self.logs, exist_ok=True)
        self.writer = SummaryWriter(self.logs)

        # Save train parameters
        with open('{}/eval_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(eval_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset_configs/{self.dataset}.yml')
        self.datasetname = dataset_configs.datasetname
        self.root_dir = dataset_configs.root_dir
        self.test_dir = dataset_configs.val_dir
        self.class_name = dataset_configs.cls
        self.class_w = dataset_configs.cls_w
        self.num_class = len(self.class_name)
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std

        # Data Loader
        self.device = eval_opt.device
        self.img_size = eval_opt.img_size
        self.batch_size = eval_opt.batch_size


        eval_transforms = [
            tr.Normalizer(with_std=False, mean=self.mean, std=self.std),
            # tr.Resizer(img_size=self.img_size, mean_padding=True, mean=self.mean)
            tr.Resizer_cv2(img_size=self.img_size)
        ]

        eval_set = load_visionin_wood(root_dir=self.test_dir, class_name=self.class_name,
                                     transform=transforms.Compose(eval_transforms))

        val_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': eval_opt.num_worker
        }
        self.eval_generator = DataLoader(eval_set, **val_params)

        # Model
        if self.model_name == 'u2net':
            model = U2NET(out_ch=self.num_class)
        elif self.model_name == 'u2netp':
            model = U2NETP(out_ch=self.num_class)
        elif self.model_name == 'unet':
            model = UNet(n_classes=self.num_class, bilinear=False)
        elif self.model_name == 'deeplabv3':
            model = DeepLabV3_ly(out_ch=self.num_class, pretrain=False)
        elif self.model_name == 'segmenter':
            model = Segmenter_Ly(img_size=self.img_size, patch_size=8, depth=12, embed_dim=192, decoder_dim=192, mlp_ratio=4, n_heads=3, in_chans=2, out_chans=self.num_class)
        else:
            model = VitSegVisionin(backbone=self.model_name, img_size=self.img_size, in_ch=2, out_ch=self.num_class)

        assert eval_opt.ckpt is not None

        weight = torch.load(eval_opt.ckpt, map_location=self.device)
        model.load_state_dict(weight, strict=True)

        self.model = ModelWithLoss(model=model, loss_func_type=eval_opt.loss_type, reduction='mean', class_w=None)
        self.model = self.model.to(self.device)

        self.num_iter_per_epoch = len(self.eval_generator)
        self.offset = 1e-6
        self.cms_cls = None

    def eval(self):
        self.model.eval()
        progress_bar = tqdm(self.eval_generator)
        losses = []
        ious = []
        self.cms_cls = np.zeros((self.num_class, 2, 2))

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    imgs, mask = data['img'], data['mask']
                    if len(imgs.shape) == 3:
                        imgs = torch.unsqueeze(imgs, dim=1)
                    else:
                        imgs = imgs.permute(0, 3, 1, 2)
                    imgs = imgs.to(self.device)

                    if len(mask.shape) == 3:
                        mask = torch.unsqueeze(mask, dim=1)
                    else:
                        mask = mask.permute(0, 3, 1, 2)
                    mask = mask.to(self.device)

                    loss, cm_cls = self.model(imgs, mask)

                    self.cms_cls += cm_cls

                    sum_cm = np.sum(cm_cls, axis=0)

                    TN = sum_cm[0][0]
                    FN = sum_cm[1][0]
                    TP = sum_cm[1][1]
                    FP = sum_cm[0][1]

                    iou = TP / (TP + FP + FN + self.offset)
                    ious.append(iou)
                    losses.append(loss.item())

                    descriptor = '[Eval] Iteration: {}/{}. Loss: {}. IoU: {}'.format(iter + 1, len(progress_bar), loss, iou)
                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        mLoss = np.mean(losses)
        mIoU = np.mean(ious)
        val_descrip = '[Eval] Mean Dice Loss: {}. Mean IoU: {}'.format(mLoss, mIoU)
        print(val_descrip)

    def start(self):
        self.eval()
        self.eval_by_metrics()

    def data_measurement(self):
        progress_bar = tqdm(self.eval_generator)
        for iter, data in enumerate(progress_bar):
            imgs, mask = data['img'], data['mask']
            if len(imgs.shape) == 3:
                imgs = torch.unsqueeze(imgs, dim=1)
            else:
                imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(self.device)

            if len(mask.shape) == 3:
                mask = torch.unsqueeze(mask, dim=1)
            else:
                mask = mask.permute(0, 3, 1, 2)

    def eval_by_metrics(self):
        if self.cms_cls is not None:
            cm = np.sum(self.cms_cls, axis=0)[:self.num_class-1]
            TN = cm[0][0]
            FN = cm[1][0]
            TP = cm[1][1]
            FP = cm[0][1]
            iou = TP / (TP + FP + FN + self.offset)
            precision = TP / (TP + FP + self.offset)
            recall = TP / (TP + FN + self.offset)
            acc = (TP + TN) / (TP + TN + FP + FN + self.offset)
            print("Overall:")
            # print(cm)
            print("IoU: {} - P: {} - R: {} - Acc: {}".format(iou, precision, recall, acc))
            for idx in range(self.num_class -1):
                cm = self.cms_cls[idx, :, :]
                TN = cm[0][0]
                FN = cm[1][0]
                TP = cm[1][1]
                FP = cm[0][1]
                iou = TP / (TP + FP + FN + self.offset)
                precision = TP / (TP + FP + self.offset)
                recall = TP / (TP + FN + self.offset)
                acc = (TP + TN) / (TP + TN + FP + FN + self.offset)
                print("Class {}".format(idx))
                # print(cm)
                print("IoU: {} - P: {} - R: {} - Acc: {}".format(iou, precision, recall, acc))
