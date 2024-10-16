import math

import cv2
import pandas as pd

from src.model_prediction import U2Net_Prediction
from utils.utils import YamlRead
import os
import numpy as np
from tqdm import tqdm


nc = 5
color_palette = np.random.uniform(0, 255, size=(nc, 3))


def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # assert iou >= 0.0
    # assert iou <= 1.0
    # return iou
    return intersection_area / bb2_area


def contourIntersect(original_image, contour1, contour2, fn=None):
    # Two separate contours trying to check intersection on
    # contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    # image1 = cv2.drawContours(blank.copy(), [contours[0]], 0, 1)
    # image2 = cv2.drawContours(blank.copy(), [contours[1]], 1, 1)
    image1 = cv2.drawContours(blank.copy(), [contour1], 0, 1, thickness=cv2.FILLED)
    image2 = cv2.drawContours(blank.copy(), [contour2], 0, 1, thickness=cv2.FILLED)

    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    # return intersection.any()
    # if fn == '3':
    #     print(1)
    if not intersection.any():
        return False
    iou_point = np.where(intersection==True)
    penmart_point = np.where(image1==1)
    if len(iou_point[0]) / len(penmart_point[0]) >= 0.1:
        return True
    else:
        return False


def predict_img(model, image, onnx=False, dataset='visionin_wood', fn=None, getBox=False):
    dataset_configs = YamlRead(f'configs/db_configs/{dataset}.yml')
    nc = dataset_configs.num_cls
    if onnx:
        o = model.prediction_onnx(image)
    else:
        o = model(image)
    len_idx = [0, 0, 0]
    h, w, _ = image.shape
    center_rotor = []
    center_stator = []
    center_penmark = []
    boxes = []
    ids = []
    for cls in range(0, nc):
        color = color_palette[cls]
        w_h = o[:, :, cls]
        w_h *= 255
        contours, _ = cv2.findContours(w_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for i in range(len(contours)):
            # hull = cv2.convexHull(contours[i])
            hull = contours[i]
            hull_list.append(hull)
            image = cv2.drawContours(image, hull_list, i, color, 2)
            (xb, yb, wb, hb) = cv2.boundingRect(hull)
            x2 = xb + wb
            y2 = yb + hb
            box = [xb, yb, x2, y2]
            if wb > 10 and hb > 10:
                boxes.append(box)
                ids.append(cls)
            # if wb > 5 and hb > 5:
            # if cls == 2:
            #     cv2.rectangle(image, (xb, yb), (xb + wb, yb + hb), color, 1)
    if getBox:
        return image, np.array(boxes), np.array(ids)
    return image


def predict_folder(inFolder, outFolder, model, onnx=False, dataset='visionin_samsung_ruler', get_rs=None):
    all_files = os.listdir(inFolder)
    fname = []
    results = []
    # if train_dir is not None:
    # train_list = os.listdir(train_dir)
    for f in tqdm(all_files):
        if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.do')):
            img_path = os.path.join(inFolder, f)
            fn = os.path.splitext(f)[0]
            fname.append(f)
            # out_path = os.path.join(outFolder, '{}.png'.format(fn))
            img = cv2.imread(img_path)
            o, bboxes, class_ids = predict_img(model, img, onnx=onnx, dataset=dataset, fn=fn, getBox=True)
            # cv2.imwrite(out_path, o)
            predict = zip(bboxes, class_ids)
            for box, id in predict:
                x, y, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cl = (0, 255, 0)
                if id == 2:
                    cv2.rectangle(o, (x, y), (x1, y1), cl, 1)
            if len(class_ids):
                det_cls = class_ids.tolist()
                num_safe_belt = det_cls.count(1)
                num_hook = det_cls.count(2)
                num_cable = det_cls.count(3)
                if num_hook > 1:
                    result = 'Pass'
                    reason = 'Hook visible > 1'
                    if num_safe_belt > 0:
                        index = np.where(class_ids == 1)
                        belt_boxes = bboxes[index]
                        x_min = min(belt_boxes[:, 0])
                        y_min = min(belt_boxes[:, 1])
                        x_max = max(belt_boxes[:, 2])
                        y_max = max(belt_boxes[:, 3])
                        cv2.rectangle(o, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

                elif num_hook == 1 and num_safe_belt > 0:
                    index = np.where(class_ids == 1)
                    belt_boxes = bboxes[index]
                    x_min = min(belt_boxes[:, 0])
                    y_min = min(belt_boxes[:, 1])
                    x_max = max(belt_boxes[:, 2])
                    y_max = max(belt_boxes[:, 3])

                    cv2.rectangle(o, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

                    belt_box = [x_min, y_min, x_max, y_max]

                    index = np.where(class_ids == 2)
                    filter_box = bboxes[index]

                    result = 'Fail'
                    reason = 'No Hook'

                    for hook_box in filter_box:
                        iou_box2_percent = get_iou(belt_box, hook_box)
                        if iou_box2_percent <= 0.5:
                            result = 'Pass'
                            reason = 'Hook visible'
                            break
                elif num_hook == 1 and num_safe_belt == 0:
                    result = 'Pass'
                    reason = 'external Hook visible'
                else:
                    result = 'Fail'
                    reason = 'No Hook'
            else:
                result = 'Fail'
                reason = 'No Hook'
            results.append(result)

            if 'Fail' in result:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            img = cv2.putText(img, result, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            img = cv2.putText(img, reason, (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            save_f = os.path.join(outFolder, f)
            cv2.imwrite(save_f, img)

    header = ['fn', 'predict']
    save_name = os.path.join(outFolder, 'predict.csv')
    df = pd.DataFrame(zip(fname, results), dtype=str, columns=header)
    df.to_csv(save_name, index=False)



def measure(label, predict):
    df_label = pd.read_csv(label, encoding='utf-8')
    # lb_dic = {'준수': 'Pass', '미준수': 'Fail'}
    lb_dic = {'o': 'Pass', 'O': 'Pass', 'x': 'Fail', 'X': 'Fail'}
    df_predict = pd.read_csv(predict, encoding='utf-8', keep_default_na=False)
    cf_matrix = np.zeros((2, 2))
    fp_cases = []
    fn_cases = []
    for row in df_label.iterrows():
        number = row[1]['fn']
        lb_name_old = row[1]['hook']
        lb_name = row[1]['Change']
        if isinstance(lb_name, float):
            lb_name = lb_name_old
        img_name = '{}.jpg'.format(number)
        if img_name not in df_predict['fn'].tolist():
            continue
        pd_idx = df_predict['fn'][df_predict['fn'] == img_name].index.tolist()[0]
        pd_name = df_predict.iloc[pd_idx]['predict']
        # list(dictionary).index('test')
        if lb_name == 'o' or lb_name == 'O':
            if lb_dic[lb_name] == pd_name:
                cf_matrix[0, 0] += 1
            else:
                cf_matrix[0, 1] += 1
                fn_cases.append(number)
        else:
            if lb_dic[lb_name] == pd_name:
                cf_matrix[1, 1] += 1
            else:
                cf_matrix[1, 0] += 1
                fp_cases.append(number)

    print(cf_matrix)
    tp = cf_matrix[0, 0]
    tn = cf_matrix[1, 1]
    fn = cf_matrix[0, 1]
    fp = cf_matrix[1, 0]
    acc = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Hook Acc: {}".format(tp / (tp + fn)))
    print("No Hook Acc: {}".format(tn / (tn + fp)))
    print("mAcc: {}".format(acc))
    print("Pre: {}".format(precision))
    print("Re: {}".format(recall))
    print("FP: {}".format(fp_cases))
    print("FN: {}".format(fn_cases))

if __name__ == '__main__':
    ckpt = r'D:\visionin\workspace\src_code\github_repo\runs\train\VisioninSamsungTask6\u2net_visionin_samsung_task6\single_run\2024.09.10_14.45.39\last.pt'
    device = 'cuda'
    img_size = 512
    th = 0.2
    nc = 5
    maxArea = False
    dataset = 'visionin_samsung_task6'
    model = U2Net_Prediction(ckpt=ckpt, img_size=img_size, nc=nc, th=th, device=device, custom=None)
    root = r'D:\visionin\workspace\datasets\samsung_house\Samsung\task6'
    img_f = 'hook_seg'
    rs_f = 'rs_seg'

    inf = os.path.join(root, img_f)
    outf = os.path.join(root, rs_f)

    predict_folder(inf, outf, model, onnx=False, dataset=dataset, get_rs=True)

    root = r'D:\visionin\workspace\datasets\samsung_house\Samsung\task6\rs_seg'
    label = os.path.join(root, 'gt_hook.csv')
    print("Label file: gt_hook.csv")
    predict = os.path.join(root, 'predict.csv')
    measure(label, predict)
