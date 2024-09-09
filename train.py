# This is a train Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import argparse
# from src.trainer import Trainer
from src.trainer import Trainer
from src.trainer_vit_bert import Trainer as Trainer_prompt
from utils.utils import Dict2Class
import nni


def get_args():
    parser = argparse.ArgumentParser('Segmentation Pytorch')
    parser.add_argument('-p', '--project', type=str, default='VisioninSamsungTask6', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='Swin-B', help='Choosing Model: Swin-B, u2net')
    parser.add_argument('-c', '--ckpt', type=str, default=None, help='Load pretrained weight')
    parser.add_argument('-d', '--dataset', type=str, default='visionin_samsung_task6', help='Choosing Dataset')
    parser.add_argument('-lr', '--lr', type=float, default=2e-4, help='Init Learning Rate')
    parser.add_argument('-ep', '--epochs', type=int, default=990, help='Init number of train epochs')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Init train batch size')
    parser.add_argument('-is', '--img_size', type=int, default=512, help='Init image size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-lt', '--loss_type', type=str, default='dice', help='Choosing loss function')
    parser.add_argument('-op', '--optimizer', type=str, default='adamw', help='Choosing optimizer')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default='cosine', help='Choosing learning rate scheduler')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    opt = get_args()

    nni_params = nni.get_next_parameter()
    if len(nni_params):
        opt.update(nni_params)
        run_nni = {"nni": True}
        opt.update(run_nni)

    opt = Dict2Class(opt)
    trainer = Trainer(opt)
    trainer.start()
