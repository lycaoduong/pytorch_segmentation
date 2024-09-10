import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import os
import json
from labelme.utils import shape_to_mask
import base64
from PIL import Image
import io


class load_from_csv(Dataset):
    def __init__(self, csv_dir, img_dir, nc, transform=None):
        self.annotations = pd.read_csv(csv_dir)
        self.root_dir = img_dir
        self.nc = nc
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        data_loader = {'img': image, 'mask': mask}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_image(self, idx):
        img_dir = os.path.join(self.root_dir, 'image', self.annotations.iloc[idx, 0] + '.jpg')
        img = cv2.imread(img_dir)
        return img

    def load_mask(self, idx):
        mask_dir = os.path.join(self.root_dir, 'label', self.annotations.iloc[idx, 0] + '_segmentation.png')
        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        melanoma = self.annotations.iloc[idx, 1]
        keratosis = self.annotations.iloc[idx, 2]
        if melanoma == 1:
            mask *= 1.0
        elif keratosis == 1:
            mask *= 2.0
        else:
            mask *= 0.5
        one_hot_targets = self.onehot_initialization(mask, self.nc)
        return one_hot_targets

    def onehot_initialization(self, targets, nc):
        ncols = nc
        out = np.zeros(targets.shape + (ncols,), dtype=float)
        for i in range(nc):
            if np.max(targets) == 0.5:
                out[:, :, i] = np.where(targets == 0.5, 0.5, 0.0)
            else:
                out[:, :, i] = np.where(targets == i + 1, 1.0, 0.0)
        return out


class load_data_root_dir(Dataset):
    def __init__(self, root_dir, class_name={0: 'fire'}, transform=None, **kwargs):

        self.img_dir = os.path.join(root_dir, 'image')
        self.list_image = os.listdir(self.img_dir)
        self.label_dir = os.path.join(root_dir, 'label')
        self.transform = transform

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        data_loader = {'img': image, 'mask': mask}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_image(self, idx):
        img_path = os.path.join(self.img_dir, self.list_image[idx])
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        return img

    def load_mask(self, idx):
        img_path = os.path.join(self.label_dir, self.list_image[idx])
        img_path = img_path[:-4] + '.png'
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        return mask


class load_data_hyudai_v3(Dataset):
    def __init__(self, root_dir, transform=None, **kwargs):

        self.label_dir = os.path.join(root_dir, 'label')
        self.list_image = os.listdir(self.label_dir)

        self.img1_dir = os.path.join(root_dir, 'image1')
        self.img2_dir = os.path.join(root_dir, 'image2')

        self.transform = transform

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, idx):
        image, mask, prompt = self.load_img_label(idx)
        data_loader = {'img': image, 'mask': mask, 'prompt': prompt}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_img_label(self, idx):
        mask_name = self.list_image[idx]
        prompt = mask_name[3:-4]
        img_name = '{}.png'.format(mask_name[:2])
        mask = cv2.imread(os.path.join(self.label_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        image1 = cv2.imread(os.path.join(self.img1_dir, img_name), cv2.IMREAD_GRAYSCALE)
        image1 = np.expand_dims(image1, axis=2)
        image2 = cv2.imread(os.path.join(self.img2_dir, img_name), cv2.IMREAD_GRAYSCALE)
        image2 = np.expand_dims(image2, axis=2)
        image = np.concatenate((image1, image2, image2), axis=2)
        return image, mask, prompt


class load_data_hyudai_v2(Dataset):
    def __init__(self, root_dir, class_name={'fire': 1, 'smoke': 2, 'human': 3}, transform=None, db=False):
        self.img_dir = os.path.join(root_dir, 'image')
        self.json_dir1 = os.path.join(root_dir, 'label1')
        self.json_dir2 = os.path.join(root_dir, 'label2')
        self.json_list = os.listdir(self.json_dir1)
        self.transform = transform
        self.cls_name = class_name
        self.db = db
        self.color_dic= np.array([[0,   0,   0],
                                [241,  90, 255],
                                [158,  7,  66],
                                [64, 243,   0],
                                [232,  46,  62],
                                [59,  30, 247],
                                [67, 250, 250],
                                [141,  30, 200],
                                [153, 255, 255],
                                [255, 171, 248],
                                [255, 182,   0],
                                [233,   0, 255],
                                [255,  30, 100],
                                [181, 161,   9],
                                [98, 180,  80],
                                [176,  78, 118],
                                [181, 141, 255],
                                [255, 207,   0],
                                [141,  59, 134],
                                [81,  28, 248],
                                [123, 188, 255],
                                [0, 100, 255],
                                [141, 255, 152],
                                [24, 165, 123],
                                [145, 163,106],
                                [41,  90, 241],
                                [197, 233,   0],
                                [69, 20, 254],
                                [155, 230,  50],
                                [179, 153, 136],
                                [76,   0, 153],
                                [110, 234, 247]])

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        image, mask = self.load_img_label(idx)
        data_loader = {'img': image, 'mask': mask, 'prompt': 0.0}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_img_label(self, idx):
        file_name = self.json_list[idx]
        tem_idx = int(file_name[:-5])
        label_1_p = os.path.join(self.json_dir1, '{}.json'.format(tem_idx))
        label_2_p = os.path.join(self.json_dir2, '{}.json'.format(tem_idx+1))
        image_1_p = os.path.join(self.img_dir, '{}.bmp'.format(tem_idx))
        image_2_p = os.path.join(self.img_dir, '{}.bmp'.format(tem_idx+1))

        data1 = json.load(open(label_1_p))
        data2 = json.load(open(label_2_p))
        image_1 = cv2.imread(image_1_p)
        image_2 = cv2.imread(image_2_p)
        image = np.zeros_like(image_1)
        image[:, :, 0] = image_1[:, :, 0]
        image[:, :, 1] = image_2[:, :, 0]

        #mask_label = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_label = np.zeros(image.shape[:2] + (len(self.cls_name),), dtype=float)

        if len(data1['shapes']):
            for shape in data1['shapes']:
                points = shape['points']
                label = shape['label']
                shape_type = shape["shape_type"]
                if shape_type=='polygon':
                    mask = shape_to_mask(image.shape[:2], points, shape_type)
                    # mask_label[mask] = int(self.cls_name[label])
                    mask_label[:, :, int(self.cls_name[label])-1][mask] = 1.0

        if len(data2['shapes']):
            for shape in data2['shapes']:
                points = shape['points']
                label = shape['label']
                shape_type = shape["shape_type"]
                if shape_type=='polygon':
                    mask = shape_to_mask(image.shape[:2], points, shape_type)
                    # mask_label[mask] = int(self.cls_name[label])
                    mask_label[:, :, int(self.cls_name[label]) - 1][mask] = 1.0

        if self.db:
            sum_mask = np.sum(mask_label, axis=2).astype(np.uint8)
            mask = self.color_dic[sum_mask].astype(np.uint8)
            db_img = cv2.addWeighted(image_2, 0.7, mask, 0.3, 0.0)
            cv2.imwrite('mask_db.png', db_img)

        # label_mask = self.onehot_encoder(mask_label, len(self.cls_name))
        return image[:, :, :2], mask_label

    def onehot_encoder(self, targets, nc):
        ncols = nc
        out = np.zeros(targets.shape + (ncols,), dtype=float)
        for i in range(nc):
            out[:, :, i] = np.where(targets == i + 1, 1.0, 0.0)
        return out


class load_visionin_wood(Dataset):
    def __init__(self, root_dir, class_name={'fire': 1, 'smoke': 2, 'human': 3}, transform=None, db=False, nc=None):
        # self.json_list = [f for f in os.listdir(root_dir) if f.endswith('.json')]
        self.img_list = []
        for f in os.listdir((root_dir)):
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                self.img_list.append(f)
        self.root_dir = root_dir
        self.transform = transform
        self.cls_name = class_name
        if nc is None:
            self.nc = len(class_name)
        else:
            self.nc = nc
        self.db = db
        self.color_dic = np.array([[0,   0,   0],
                                [241,  90, 255],
                                [158,  7,  66],
                                [64, 243,   0],
                                [232,  46,  62],
                                [59,  30, 247],
                                [67, 250, 250],
                                [141,  30, 200],
                                [153, 255, 255],
                                [255, 171, 248],
                                [255, 182,   0],
                                [233,   0, 255],
                                [255,  30, 100],
                                [181, 161,   9],
                                [98, 180,  80],
                                [176,  78, 118],
                                [181, 141, 255],
                                [255, 207,   0],
                                [141,  59, 134],
                                [81,  28, 248],
                                [123, 188, 255],
                                [0, 100, 255],
                                [141, 255, 152],
                                [24, 165, 123],
                                [145, 163,106],
                                [41,  90, 241],
                                [197, 233,   0],
                                [69, 20, 254],
                                [155, 230,  50],
                                [179, 153, 136],
                                [76,   0, 153],
                                [110, 234, 247]])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image, mask = self.load_img_label(idx)
        data_loader = {'img': image, 'mask': mask, 'prompt': 0.0}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_img_label(self, idx):
        image_name = self.img_list[idx]
        image_p = os.path.join(self.root_dir, image_name)
        label_p = '{}.json'.format(os.path.splitext(image_p)[0])

        image = cv2.imread(image_p)

        mask_label = np.zeros(image.shape[:2] + (self.nc,), dtype=float)

        if os.path.isfile(label_p):
            data = json.load(open(label_p))
            if len(data['shapes']):
                for shape in data['shapes']:
                    points = shape['points']
                    label = shape['label']
                    shape_type = shape["shape_type"]
                    if label in self.cls_name:
                        if shape_type == 'polygon':
                            mask = shape_to_mask(image.shape[:2], points, shape_type)
                            mask_label[:, :, int(self.cls_name[label]) - 1][mask] = 1.0
                            # mask_label[mask] = int(self.cls_name[label])
                        if shape_type == 'mask':
                            mask_b64 = shape['mask']
                            img_bytes = base64.b64decode(mask_b64.encode('utf-8'))
                            img = Image.open(io.BytesIO(img_bytes))
                            img_arr = np.asarray(img)
                            x1, y1 = int(points[0][0]), int(points[0][1])
                            x2, y2 = int(points[1][0]), int(points[1][1])
                            mask_label[y1:y2 + 1, x1:x2 + 1, int(self.cls_name[label]) - 1] = img_arr
                    # if shape_type == 'polygon':
                    #     if 'top' in label:
                    #         mask = shape_to_mask(image.shape[:2], points, shape_type)
                    #         mask_label[:, :, 0][mask] = 1.0
                    #     elif 'side' in label:
                    #         mask = shape_to_mask(image.shape[:2], points, shape_type)
                    #         mask_label[:, :, 1][mask] = 1.0
                    #     elif 'water' in label:
                    #         mask = shape_to_mask(image.shape[:2], points, shape_type)
                    #         mask_label[:, :, 2][mask] = 1.0


        if self.db:
            sum_mask = np.sum(mask_label, axis=2).astype(np.uint8)
            # sum_mask = mask_label[:,:,1].astype(np.uint8)
            mask = self.color_dic[sum_mask].astype(np.uint8)
            db_img = cv2.addWeighted(image, 0.7, mask, 0.3, 0.0)
            cv2.imwrite('mask_db.png', db_img)

        # label_mask = self.onehot_encoder(mask_label, len(self.cls_name))
        return image, mask_label


class load_data_json_nia(Dataset):
    def __init__(self, root_dir, cls_dic, set_name='train', transform=None, **kwargs):

        if set_name == 'train':
            self.total_data = [line.rstrip('\n') for line in open(os.path.join(root_dir, 'train.txt'))]
        else:
            self.total_data = [line.rstrip('\n') for line in open(os.path.join(root_dir, 'val.txt'))]
        self.transform = transform
        self.cls_dic = cls_dic
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, '220818_NIA_35_raw_img_data_vision-in')
        self.label_dir = os.path.join(root_dir, '220818_NIA_35_json_data_vision-in')
        self.num_class = len(cls_dic)

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        image, mask = self.load_image_mask(idx)
        data_loader = {'img': image, 'mask': mask}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_image_mask(self, idx):
        path = self.total_data[idx]
        imgpath = os.path.join(self.img_dir, path)
        img = cv2.imread(imgpath)
        annot_path = self.label_dir + '/' + path[:-3] + 'json'
        mask = self.load_mask(annot_path)
        return img, mask

    def load_mask(self, annot_path):
        data = json.load(open(annot_path))
        annot_info = data.get('information')
        if annot_info:
            # file_name = annot_info['filename']
            (img_w, img_h) = annot_info['resolution']
            mask = np.zeros((img_h, img_w, 3), np.uint8)
            annotations = data.get('annotations')
            for annot in annotations:
                class_name = annot['class'].lower()
                if class_name != 'background':
                    label_value = self.cls_dic[class_name]
                    polygon = np.array(annot['polygon'])
                    contour = polygon.reshape((int(len(polygon) / 2), -1))
                    mask = cv2.fillPoly(mask, pts=[contour], color=(label_value, label_value, label_value))
            mask = mask[:, :, 0]
            one_hot = self.onehot_encoder(mask, nc=self.num_class)
            return one_hot

    def onehot_encoder(self, targets, nc):
        ncols = nc
        out = np.zeros(targets.shape + (ncols,), dtype=float)
        for i in range(nc):
            out[:, :, i] = np.where(targets == i + 1, 1.0, 0.0)
        return out


if __name__ == '__main__':
    print(1)


