import datetime
import os
import json
from utils.utils import YamlRead
from utils import transform as tr
from torchvision import transforms
from utils.data_loader import load_data_hyudai_v3
from torch.utils.data import DataLoader
from networks.u2net.u2net import U2NET, U2NETP
from networks.VitSeg_Visionin.vitseg_visionin import VitSegVisionin
from networks.deeplabv3.deeplabv3 import DeepLabV3_ly
from networks.transformer.transformer_img import Vit_Bert
from networks.clipseg.clipseg import CLIPDensePredTMasked_ly
from torch import nn
from utils.loss_function import Custom_Loss, iou_calculator_ly
from utils.optimizer import SharpnessAwareMinimization
import torch
from tqdm.autonotebook import tqdm
import traceback
from tensorboardX import SummaryWriter
import numpy as np
import nni
from transformers import BertTokenizer


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_func_type='dice', reduction='mean', class_w=None):
        super().__init__()
        self.model = model
        self.criterion = Custom_Loss(loss_type=loss_func_type, reduction=reduction)
        self.weight = class_w

    # def forward(self, image, mask, **kwargs):
    #     output = self.model(image)
    #     losses = self.criterion(output, mask, self.weight)
    #     iou = iou_calculator_ly(output, mask)
    #     return losses, iou

    # def forward(self, image, prompt_ebbed, target, lb_mask, **kwargs):
    #     output = self.model(image, prompt_ebbed, target, lb_mask)
    #     losses = self.criterion(output, target, self.weight)
    #     iou = iou_calculator_ly(output, target)
    #     return losses, iou

    def forward(self, image, prompt, target, **kwargs):
        output = self.model(image, prompt)
        losses = self.criterion(output, target, self.weight)
        iou = iou_calculator_ly(output, target)
        return losses, iou


class Trainer(object):
    def __init__(self, train_opt):
        self.project = train_opt.project
        self.model_name = train_opt.model
        self.dataset = train_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)

        if hasattr(train_opt, 'nni'):
            # exp_name, _, trial_name = os.environ['NNI_OUTPUT_DIR'].split('\\')[-3:]
            exp_name, _, trial_name = os.environ['NNI_OUTPUT_DIR'].split('/')[-3:]
            self.nni_writer = True
        else:
            exp_name, trial_name = 'single_run', date_time
            self.nni_writer = False
        self.save_dir = '../runs/train/{}/{}_{}/{}/{}/'.format(self.project, self.model_name, self.dataset, exp_name, trial_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.logs = self.save_dir + 'logs/'
        os.makedirs(self.logs, exist_ok=True)
        self.writer = SummaryWriter(self.logs)

        # Save train parameters
        with open('{}/train_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(train_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset_configs/{self.dataset}.yml')
        self.datasetname = dataset_configs.datasetname
        self.root_dir = dataset_configs.root_dir
        self.train_dir = dataset_configs.train_dir
        self.val_dir = dataset_configs.val_dir
        self.class_name = dataset_configs.cls
        self.class_w = dataset_configs.cls_w
        self.num_class = len(self.class_name)
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std

        # Data Loader
        self.device = train_opt.device
        self.img_size = train_opt.img_size
        self.batch_size = train_opt.batch_size

        train_transforms = [
            tr.Custom_Augment(
                gray_scale=0.0,
                rotate=0.15,
                scale=0.0,
                scale_range=[0.7, 1.3, 0.1],
                flipx=0.2,
                flipy=0.2,
                brightness=0.0,
                colorjitter=0.0,
                blur=0.0,
                eqHis=0.0,
                rd_crop=0.0
            ),
            tr.Normalizer(with_std=False, mean=self.mean, std=self.std),
            # tr.Resizer(img_size=self.img_size, mean_padding=True, mean=self.mean)
            tr.Resizer_cv2(img_size=self.img_size)
        ]

        # training_set = load_from_csv(csv_dir=self.train_dir, img_dir=self.root_dir, nc=self.num_class, transform=transforms.Compose(train_transforms))
        training_set = load_data_hyudai_v3(root_dir=self.train_dir, transform=transforms.Compose(train_transforms))
        #training_set = load_data_json_nia(root_dir=self.root_dir, cls_dic=self.class_name, set_name='train', transform=transforms.Compose(train_transforms))

        train_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': train_opt.num_worker
        }

        self.training_generator = DataLoader(training_set, **train_params)

        validation_transforms = [
            tr.Normalizer(with_std=False, mean=self.mean, std=self.std),
            # tr.Resizer(img_size=self.img_size, mean_padding=True, mean=self.mean)
            tr.Resizer_cv2(img_size=self.img_size)
        ]

        # val_set = load_from_csv(csv_dir=self.val_dir, img_dir=self.root_dir, nc=self.num_class,
        #                              transform=transforms.Compose(validation_transforms))
        val_set = load_data_hyudai_v3(root_dir=self.val_dir, transform=transforms.Compose(validation_transforms))
        #val_set = load_data_json_nia(root_dir=self.root_dir, cls_dic=self.class_name, set_name='val', transform=transforms.Compose(validation_transforms))

        val_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': train_opt.num_worker
        }
        self.val_generator = DataLoader(val_set, **val_params)

        # Model
        if self.model_name == 'u2net':
            model = U2NET(in_ch=2, out_ch=self.num_class)
        elif self.model_name == 'u2netp':
            model = U2NETP(in_ch=2, out_ch=self.num_class)
        elif self.model_name == 'deeplabv3':
            model = DeepLabV3_ly(in_ch=2, out_ch=self.num_class, pretrain=False)
        elif self.model_name == 'clipseg':
            model = CLIPDensePredTMasked_ly()
        elif self.model_name == 'vit_bert':
            self.eb_dim = 192
            self.path_size = 8
            model = Vit_Bert(img_size=self.img_size, patch_size=self.path_size, in_chans=2, n_heads=3, embed_dim=self.eb_dim)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            path_dim = int(self.img_size / self.path_size)
            self.lb_mask = self.src_mask(path_dim * path_dim).to(self.device)
        else:
            model = VitSegVisionin(backbone=self.model_name, img_size=self.img_size, in_ch=2, out_ch=self.num_class)

        if train_opt.ckpt is not None:
            weight = torch.load(train_opt.ckpt, map_location=self.device)
            model.load_state_dict(weight, strict=True)

        # if torch.cuda.is_available():
        #     model = nn.DataParallel(model)
        # NGPU = torch.cuda.device_count()
        # if NGPU > 1:
        #     model = nn.DataParallel(model)

        self.model = ModelWithLoss(model=model, loss_func_type=train_opt.loss_type, reduction='mean', class_w=None)
        self.model = self.model.to(self.device)

        # Optimizer and Learning rate scheduler

        self.opti = train_opt.optimizer
        self.l_rate = train_opt.lr
        self.lr_scheduler = train_opt.lr_scheduler

        if self.opti == 'sam':
            base_optimizer = torch.optim.SGD
            self.optimizer = SharpnessAwareMinimization(params=self.model.parameters(),
                                                        base_optimizer=base_optimizer,
                                                        lr=self.l_rate,
                                                        momentum=0.9)
        elif self.opti == 'adamw':
            self.optimizer = torch.optim.Adam(params=model.parameters(),
                                              lr=self.l_rate)
        else:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.l_rate,
                                             momentum=0.9)

        if self.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                                  T_0=10,
                                                                                  T_mult=2)
        elif self.lr_scheduler == 'lambdalr':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        self.num_iter_per_epoch = len(self.training_generator)
        self.step = 0
        self.best_loss = 1e5
        self.best_iou = 0
        self.epochs = train_opt.epochs


    def train(self, epoch):
        self.model.train()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.training_generator)
        epoch_loss = []
        epoch_iou = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                imgs, mask, prompt = data['img'], data['mask'], data['prompt']
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

                if self.model_name == 'clipseg':
                    loss, iou = self.model(imgs, prompt, mask)
                elif self.model_name == 'vit_bert':
                    bert_input = self.tokenizer(prompt, padding='max_length', max_length=self.eb_dim,
                                           truncation=True, return_tensors="pt")
                    prompt_ebbed = bert_input['input_ids'].to(self.device)
                    prompt_ebbed = torch.unsqueeze(prompt_ebbed, dim=1)
                    prompt_ebbed = prompt_ebbed / torch.max(prompt_ebbed)
                    loss, iou = self.model(imgs, prompt_ebbed, mask, self.lb_mask)

                loss.backward()

                if self.opti == 'sam':
                    self.optimizer.first_step(zero_grad=True)
                    loss2, iou2 = self.model(imgs, mask)
                    loss2.backward()
                    self.optimizer.second_step(zero_grad=True, clip_norm=True)
                else:
                    self.optimizer.step()

                if self.lr_scheduler == 'cosine':
                    self.scheduler.step(epoch + iter / self.num_iter_per_epoch)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, self.step)

                epoch_iou.append(iou.item())
                epoch_loss.append(loss.item())
                descriptor = '[Train] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {}. IoU: {}'.format(
                        self.step, epoch+1, self.epochs, iter + 1, self.num_iter_per_epoch, loss, iou)
                progress_bar.set_description(descriptor)
                self.step += 1

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        if self.lr_scheduler == 'reduce':
            self.scheduler.step(np.mean(epoch_loss))
        elif self.lr_scheduler == 'lambdalr':
            self.scheduler.step()
        # if self.lr_scheduler == 'cosine':
        #     self.scheduler.step()

        mean_loss = np.mean(epoch_loss)
        mean_iou = np.mean(epoch_iou)
        train_descrip = '[Train] Epoch: {}. Mean Loss: {}. Mean IoU: {}'.format(epoch+1, mean_loss, mean_iou)
        print(train_descrip)
        self.writer.add_scalars('Loss', {'train': mean_loss}, epoch)
        self.writer.add_scalars('IoU', {'train': mean_iou}, epoch)

    def validation(self, epoch):
        self.model.eval()
        progress_bar = tqdm(self.val_generator)
        epoch_loss = []
        epoch_iou = []

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    imgs, mask, prompt = data['img'], data['mask'], data['prompt']
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

                    if self.model_name == 'clipseg':
                        loss, iou = self.model(imgs, prompt, mask)
                    elif self.model_name == 'vit_bert':
                        bert_input = self.tokenizer(prompt, padding='max_length', max_length=self.eb_dim,
                                                    truncation=True, return_tensors="pt")
                        prompt_ebbed = bert_input['input_ids'].to(self.device)
                        prompt_ebbed = torch.unsqueeze(prompt_ebbed, dim=1)
                        prompt_ebbed = prompt_ebbed / torch.max(prompt_ebbed)
                        loss, iou = self.model(imgs, prompt_ebbed, mask, None)

                    epoch_iou.append(iou.item())
                    epoch_loss.append(loss.item())

                    descriptor = '[Valid] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {}. Acc: {}'.format(
                        epoch * len(progress_bar) + iter, epoch, self.epochs, iter + 1, len(progress_bar), loss, iou)
                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        val_epoch_loss = np.mean(epoch_loss)
        val_epoch_iou = np.mean(epoch_iou)
        val_descrip = '[Validation] Epoch: {}. Mean Loss: {}. Mean IoU: {}'.format(epoch + 1, val_epoch_loss, val_epoch_iou)
        print(val_descrip)

        self.writer.add_scalars('Loss', {'val': val_epoch_loss}, epoch)
        self.writer.add_scalars('IoU', {'val': val_epoch_iou}, epoch)

        if self.nni_writer:
            nni.report_intermediate_result(val_epoch_iou)
            if epoch == self.epochs - 1:
                nni.report_final_result(val_epoch_iou)

        self.save_checkpoint(self.model, self.save_dir, 'last.pt')

        if self.best_loss > val_epoch_loss:
            self.best_loss = val_epoch_loss
            self.save_checkpoint(self.model, self.save_dir, 'best_val_loss.pt')

        if self.best_iou < val_epoch_iou:
            self.best_iou = val_epoch_iou
            self.save_checkpoint(self.model, self.save_dir, 'best_val_iou.pt')

    def start(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validation(epoch)

    def save_checkpoint(self, model, saved_path, name):
        torch.save(model.model.state_dict(), saved_path + name)

    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float(1.0))
        return mask
