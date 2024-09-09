import cv2
import pandas as pd

from src.model_prediction import U2Net_Prediction
from utils.utils import YamlRead
import os
import numpy as np
from tqdm import tqdm


nc = 3
color_palette = np.random.uniform(0, 255, size=(nc, 3))


def predict_img(model, image, onnx=False, dataset='visionin_wood', maxArea=False, rs=None):
    dataset_configs = YamlRead(f'configs/dataset_configs/{dataset}.yml')
    nc = dataset_configs.num_cls
    if onnx:
        o = model.prediction_onnx(image)
    else:
        o = model(image)
    len_idx = [0, 0, 0]
    box_ruler = [0, 0, 0, 0]
    result = None
    reason = None
    area_rt = 0
    h, w, _ = image.shape
    img_cnt = [[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]
    img_cnt = np.array(img_cnt)
    for cls in range(0, nc):
        color = color_palette[cls]
        w_h = o[:, :, cls]
        w_h *= 255
        contours, _ = cv2.findContours(w_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if maxArea is False:
            hull_list = []
            for i in range(len(contours)):
                # hull = cv2.convexHull(contours[i])
                hull = contours[i]
                hull_list.append(hull)
                image = cv2.drawContours(image, hull_list, i, color, -1)
                (xb, yb, wb, hb) = cv2.boundingRect(hull)
                if wb > 5 and hb > 5:
                    max_len = max(wb, hb)
                    len_idx[cls] = max_len
                    cv2.rectangle(image, (xb, yb), (xb + wb, yb + hb), color, 2)
        else:
            areas = [cv2.contourArea(c) for c in contours]
            if len(areas):
                max_index = np.argmax(areas)
                area_rt = areas[max_index] * 100 / (w * h)
                if cls == 2:
                    th = 0.05
                else:
                    th = 0.2
                if area_rt >= th:
                    cnt = contours[max_index]
                    hull = cv2.convexHull(cnt)
                    image = cv2.drawContours(image, [hull], 0, color, 5)
                    (xb, yb, wb, hb) = cv2.boundingRect(hull)
                    if wb > 5 and hb > 5:
                        max_len = max(wb, hb)
                        len_idx[cls] = max_len
                        cv2.rectangle(image, (xb, yb), (xb + wb, yb + hb), color, 2)
                        if cls == 1:
                            box_ruler[0] = [xb, yb]
                            box_ruler[1] = [xb + wb, yb]
                            box_ruler[2] = [xb + wb, yb + hb]
                            box_ruler[3] = [xb, yb + hb]
                        # print(wb)
                        # print(hb)
                        # print(wb / hb)
    if rs is None:
        return image, rs

    if len_idx[0] == 0 and len_idx[1] == 0:
        result = "Fail"
        reason = "Side view is not visible"
        cv2.putText(image, "{}: {}".format(result, reason), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image, result

    res = 1
    for pt in box_ruler:
        res = cv2.pointPolygonTest(img_cnt, (pt[0], pt[1]), False)
        if res < 1:
            break

    if res < 1:
        result = "Fail"
        reason = "Out of View"
        cv2.putText(image, "{}: {}".format(result, reason), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image, result

    if len_idx[2] != 0:
        ratio = len_idx[1] / len_idx[2]
        if ratio > 10:
            result = "Pass"
            reason = "Ruler 2 Big"
            cv2.putText(image, "{}: {}".format(result, reason), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            result = "Fail"
            if ratio > 1:
                reason = "Ruler 2 Small"
            else:
                reason = "Side view is not visible"
            cv2.putText(image, "{}: {}".format(result, reason), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        ratio = 'NAN'
        result = "Pass"
        reason = "Ruler 1"
        cv2.putText(image, "{}: {}".format(result, reason), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(image, "[{}, {}, {}: {}]".format(len_idx[0], len_idx[1], len_idx[2], area_rt), (50, 150),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "Ratio: {}".format(ratio), (50, 250),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return image, result

def predict_folder(inFolder, outFolder, model, onnx=False, maxArea=False, dataset='visionin_samsung_ruler', get_rs=None):
    all_files = os.listdir(inFolder)
    fname = []
    rs = []
    for f in tqdm(all_files):
        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.do'):
            img_path = os.path.join(inFolder, f)
            fn = os.path.splitext(f)[0]
            out_path = os.path.join(outFolder, '{}.png'.format(fn))
            img = cv2.imread(img_path)
            o, result = predict_img(model, img, onnx=onnx, maxArea=maxArea, dataset=dataset, rs=get_rs)
            cv2.imwrite(out_path, o)
            fname.append(f)
            rs.append(result)
    if get_rs is not None:
        header = ['fn', 'predict']
        save_name = os.path.join(outFolder, 'predict.csv')
        df = pd.DataFrame(zip(fname, rs), dtype=str, columns=header)
        df.to_csv(save_name, index=False)

def measure(label, predict):
    df_label = pd.read_csv(label, encoding='utf-8')
    lb_dic = {'준수': 'Pass', '미준수': 'Fail'}
    df_predict = pd.read_csv(predict, encoding='utf-8')
    cf_matrix = np.zeros((2, 2))
    for row in df_label.iterrows():
        number = row[1]['번호']
        lb_name = row[1]['준수여부']
        img_name = '{}.jpg'.format(number)
        pd_idx = df_predict['fn'][df_predict['fn'] == img_name].index.tolist()[0]
        pd_name = df_predict.iloc[pd_idx]['predict']
        # list(dictionary).index('test')
        if lb_name == '준수':
            if lb_dic[lb_name] == pd_name:
                cf_matrix[0, 0] += 1
            else:
                cf_matrix[0, 1] += 1
        else:
            if lb_dic[lb_name] == pd_name:
                cf_matrix[1, 1] += 1
            else:
                cf_matrix[1, 0] += 1

    print(cf_matrix)
    tp = cf_matrix[0, 0]
    tn = cf_matrix[1, 1]
    fn = cf_matrix[0, 1]
    fp = cf_matrix[1, 0]
    acc = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Acc: {}".format(acc))
    print("Pre: {}".format(precision))
    print("Re: {}".format(recall))

if __name__ == '__main__':
    ckpt = r'D:\visionin\workspace\src_code\github_repo\runs\train\VisioninSamsungTask3\u2net_visionin_samsung_ruler\single_run\2024.06.20_15.04.47\best_val_loss.pt'
    device = 'cuda'
    img_size = 512
    th = 0.2
    nc = 3
    maxArea = True
    dataset = 'visionin_samsung_ruler'
    # dataset = 'visionin_samsung_task5'
    model = U2Net_Prediction(ckpt=ckpt, img_size=img_size, nc=nc, th=th, device=device, custom=None)
    # export_path = '{}.onnx'.format(ckpt[:-3])
    # model.onnx_export(export_path)
    # img_path = 'D:/visionin/workspace/datasets/visionin_nas/wood/folder1/imageSrc (1).jpg'
    # img = cv2.imread(img_path)
    # rs = predict_img(model=model, image=img, dataset=dataset, onnx=False)
    # cv2.imwrite("wood_result.png", rs)
    root = r'D:\visionin\workspace\datasets\samsung_house\Samsung\ruler\balanceruler\segmentation'
    img_f = 'test'
    rs_f = 'rs'

    inf = os.path.join(root, img_f)
    outf = os.path.join(root, rs_f)
    predict_folder(inf, outf, model, onnx=False, maxArea=maxArea, dataset=dataset, get_rs=True)

    # root = r'D:\visionin\workspace\datasets\samsung_house\Samsung\ruler\balanceruler\segmentation\rs'
    # label = os.path.join(root, 'label_ori.csv')
    # predict = os.path.join(root, 'predict.csv')
    # measure(label, predict)
