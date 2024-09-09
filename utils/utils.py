import yaml
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import torch


class YamlRead:
    def __init__(self, params_path):
        self.params = yaml.safe_load(open(params_path, encoding='utf-8').read())

    def update(self, dictionary):
        self.params = dictionary

    def __getattr__(self, item):
        return self.params.get(item, None)


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def mean_std_measure_numpy(image_dir):
    psum = np.array([0, 0, 0]).astype(np.float128)
    psum_sq = np.array([0, 0, 0]).astype(np.float128)
    count = np.array([0, 0, 0]).astype(np.int128)

    list_image = tqdm(os.listdir(image_dir))
    for file in list_image:
        image = cv2.imread(os.path.join(image_dir, file))
        image = image / 255.0
        height, width, channel = image.shape
        count += height*width
        psum += np.sum(image, axis=(0, 1))
        psum_sq += np.sum(image**2, axis=(0, 1))
    mean = psum / count
    var = (psum_sq / count) - (mean ** 2)
    std = np.sqrt(var)
    return mean, std


def onehot_encoder(array_2d):
    num_channel = array_2d.max() + 1
    one_hot = np.zeros(array_2d.shape + (num_channel,), dtype=int)
    layer_idx = np.arange(array_2d.shape[0]).reshape(array_2d.shape[0], 1)
    component_idx = np.tile(np.arange(array_2d.shape[1]), (array_2d.shape[0], 1))
    one_hot[layer_idx, component_idx, array_2d] = 1
    return one_hot


class Mean_Std_Pytorch_Loader(Dataset):

    def __init__(self,
                 directory,
                 img_size=512,
                 transform=None):
        self.directory = directory
        self.ls_images = os.listdir(self.directory)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.ls_images)

    def __getitem__(self, idx):
        # import
        path = os.path.join(self.directory, self.ls_images[idx])
        image = cv2.imread(path)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = image/255.0

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = torch.from_numpy(image).to(torch.float32)
        return image


def mean_std_measure(img_dir,
                     img_size=512,
                     device='cuda',
                     batch=64,
                     num_worker=8
                     ):
    image_dataset = Mean_Std_Pytorch_Loader(directory=img_dir, img_size=img_size)
    image_loader = DataLoader(image_dataset,
                              batch_size=batch,
                              shuffle=False,
                              num_workers=num_worker,
                              pin_memory=True)
    psum = torch.tensor([0.0, 0.0, 0.0]).to(device)
    psum_sq = torch.tensor([0.0, 0.0, 0.0]).to(device)
    for inputs in tqdm(image_loader):
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.to(device)
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    # pixel count
    psum = psum.cpu()
    psum_sq = psum_sq.cpu()
    count = len(os.listdir(path)) * 512 * 512

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def hex_to_gbr(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))[::-1]


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb



if __name__ == '__main__':
    path = 'D:/lycaoduong/dataset/visionin_fire_smoke/all_images/'
    mean_std_measure(img_dir=path, img_size=512, device='cuda', batch=128)
