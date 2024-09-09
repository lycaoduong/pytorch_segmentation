import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class Custom_Loss(nn.Module):
    def __init__(self,
                 loss_type='dice',
                 reduction='mean'):
        super(Custom_Loss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        if self.loss_type == 'dice':
            self.criterion = dice_loss_efficient
        elif self.loss_type == 'bce_dice':
            self.criterion = dice_bce_loss
        elif self.loss_type == 'focal':
            self.criterion = focal_loss
        else:
            self.criterion = bce_custom

    def forward(self,
                predict,
                target,
                weight,
                **kwargs
                ):
        reduction = self.reduction
        loss = self.criterion(
            predict=predict,
            target=target,
            reduction=reduction,
            weight=weight,
            **kwargs
        )
        return loss


def dice_bce_loss(predict,
              target,
              reduction='mean',
              smooth=1e-5
              ):
    bce = F.binary_cross_entropy_with_logits(input=predict, target=target)
    #bce = bce_custom(predict=predict, target=target, reduction='none')

    # predict = torch.sigmoid(predict)

    # intersection = (predict * target).sum(dim=(2, 3))
    # union = predict.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # dice = 2.0 * (intersection + smooth) / (union + smooth)
    # dice_loss = 1.0 - dice
    #
    # loss = bce + dice_loss
    n_classes = predict.shape[1]
    d_loss = 0.
    for c in range(n_classes):
        iflat = predict[:, c].view(-1)
        tflat = target[:, c].view(-1)
        intersection = (iflat * tflat).sum()
        d_loss += (1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))
    d_loss = d_loss/n_classes
    loss = bce + d_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def focal_loss(predict,
              target,
              reduction='mean',
              smooth=1e-5,
              weight=None
              ):
    alpha = 3
    gamma = 2
    # comment out if your model contains a sigmoid or equivalent activation layer
    #prob = torch.sigmoid(output)
    prob = torch.clamp(predict, smooth, 1.0 - smooth)

    target[target >= 0.5] = 1
    pos_mask = (target == 1).float()
    neg_mask = (target == 0).float()

    pos_weight = (pos_mask * torch.pow(1 - prob, gamma)).detach()
    pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

    neg_weight = (neg_mask * torch.pow(prob, gamma)).detach()
    # neg_loss = -alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
    neg_loss = -alpha * neg_weight * torch.log(prob)  # / (torch.sum(neg_weight) + 1e-4)
    loss = pos_loss + neg_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def dice_loss(predict,
              target,
              reduction='mean',
              smooth=1e-5
              ):
    # comment out if your model contains a sigmoid or equivalent activation layer
    # predict = torch.sigmoid(predict)
    intersection = (predict * target).sum(dim=(2, 3))
    union = predict.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice

    if reduction == 'mean':
        return dice_loss.mean()
    elif reduction == 'sum':
        return dice_loss.sum()
    else:
        return dice_loss


def dice_loss_efficient(predict,
                        target,
                        reduction='mean',
                        smooth=1e-5,
                        weight=None
                        ):
    #bce = -(target * torch.log(predict) + (1. - target) * torch.log(1 - predict)).sum(dim=(2, 3))

    n_classes = predict.shape[1]
    dice_loss = 0.
    for c in range(n_classes):
        # iflat = predict[:, c].view(-1)
        # tflat = target[:, c].view(-1)
        iflat = predict[:, c].reshape(-1)
        tflat = target[:, c].reshape(-1)
        intersection = (iflat * tflat).sum()
        union = (iflat.sum() + tflat.sum())
        if weight is not None:
            w = weight[c]
            dice_loss += w * (1 - ((2. * intersection + smooth) / (union + smooth)))
        else:
            dice_loss += (1 - ((2. * intersection + smooth) / (union + smooth)))

    if reduction == 'mean':
        return dice_loss/n_classes
    else:
        return dice_loss


def bce_custom(predict,
               target,
               reduction='mean',
               weight=None,
               smooth=1e-5
               ):
    zeros = torch.zeros_like(target)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
    bce = -(target * torch.log(predict) + (1. - target) * torch.log(1 - predict)).sum(dim=(2, 3))
    if reduction == 'mean':
        return bce.mean()
    elif reduction == 'sum':
        return bce.sum()
    else:
        return bce


def iou_calculator(predict,
                   target,
                   smooth=1e-5):
    # predict = torch.sigmoid(predict)
    # predict = (predict >= 0.5).float()
    intersection = (predict * target).sum(dim=(2, 3))
    union = predict.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = intersection / (union+smooth)
    return iou.mean()


def iou_calculator_ly(predict,
                      target,
                      th=0.5,
                      smooth=1e-5):
    # predict = torch.sigmoid(predict)
    predict = (predict >= th).float()
    n_classes = predict.shape[1]
    iou = 0.
    for c in range(n_classes):
        # iflat = predict[:, c].view(-1)
        # tflat = target[:, c].view(-1)
        iflat = predict[:, c].reshape(-1)
        tflat = target[:, c].reshape(-1)
        intersection = (iflat * tflat).sum()
        union = (iflat.sum() + tflat.sum()) - intersection
        iou += intersection / (union+smooth)

    return iou/n_classes

def confusion_matrix_ly(predict,
                         target,
                         th=0.5):
    # predict = torch.sigmoid(predict)
    predict = (predict >= th).float()
    n_classes = predict.shape[1]
    cm_classes = []
    for c in range(n_classes):
        iflat = predict[:, c].reshape(-1).cpu().numpy().astype(np.int8)
        tflat = target[:, c].reshape(-1).cpu().numpy().astype(np.int8)
        cm = confusion_matrix(iflat, tflat, labels=[0, 1])
        cm_classes.append(cm)
    cm_classes = np.array(cm_classes)
    return cm_classes
