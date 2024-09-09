from utils.utils import onehot_encoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2


def evaluation_segentation(prediction, label, smooth=1e-5):
    prediction = onehot_encoder(prediction)
    label = onehot_encoder(label)
    intersection = prediction * label
    intersection = intersection.sum(axis=(0, 1))
    union = prediction.sum(axis=(0, 1)) + label.sum(axis=(0, 1))
    area_union = union - intersection
    iou = intersection / area_union
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    return iou.mean(), dice.mean()


def confusion_maxtrix(prediction, label):
    prediction = prediction.flatten()
    label = label.flatten()
    tp = (label*prediction).sum()
    fn = label.sum() - tp
    fp = prediction.sum() - tp
    tn = len(prediction) - tp - fn - fp
    cf_matrix = np.array([[tp, fn], [fp, tn]])
    return cf_matrix


def cf_matrix_save(confusion_matrix, save_dir):
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in ["0", "1"]],
                         columns=[i for i in ["0", "1"]])
    fig = plt.figure()
    ax1 = fig.add_subplot()
    # ax1.set_ylabel('Ground Truth')
    # ax1.set_xlabel('Prediction')
    ax1.set_title('Confusion Matrix')
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    fig.savefig(save_dir + '/confusion_matrix.png')

def boxplot_save(array, save_dir, label='IoU'):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.boxplot(array, patch_artist=True, labels=[label])
    plt.savefig(save_dir + '/{}.png'.format(label))

def predict_visual(image, mask, predict):
    color_map = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255]
    ]
    ).astype(np.uint8)

    predict_c = color_map[predict]
    merge = cv2.addWeighted(image, 0.5, predict_c, 0.5, 0)

    mask_predict = mask + (predict * 2)
    mask_predict = color_map[mask_predict]
    cv2.putText(mask_predict, "Ground Truth", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(mask_predict, "Prediction", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(mask_predict, "Overlap", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    out_img = np.hstack([image, mask_predict, merge])
    return out_img
