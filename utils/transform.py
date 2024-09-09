import numpy as np
import torch
import cv2
import torch.nn.functional as F


class Normalizer(object):

    def __init__(self, with_std=False, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.with_std = with_std

    def __call__(self, sample):
        image, mask, prompt = sample['img'], sample['mask'], sample['prompt']
        if self.with_std:
            image = ((image.astype(np.float32) / 255.0) - self.mean) / self.std
        else:
            image = image.astype(np.float32) / 255.0
        # mask = mask.astype(np.float32) / 255.0
        sample = {'img': image, 'mask': mask, 'prompt': prompt}
        return sample

class Resizer(object):
    def __init__(self, img_size=512, use_offset=True, mean_padding=False, mean=48):
        self.img_size = img_size
        self.use_offset = use_offset
        self.mean_padding = mean_padding
        self.mean = np.array(mean)

    def __call__(self, sample):
        image, mask, prompt = sample['img'], sample['mask'], sample['prompt']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        # print(image.shape, mask.shape)
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)


        # new_image = np.ones((self.img_size, self.img_size, 3)) * self.mean
        if self.mean_padding:
            new_image = np.ones((self.img_size, self.img_size, image.shape[2])) * self.mean
        else: # Padding with zero
            new_image = np.zeros((self.img_size, self.img_size, image.shape[2]))

        if len(mask.shape) == 2:
            new_mask = np.zeros((self.img_size, self.img_size))
        else:
            new_mask = np.zeros((self.img_size, self.img_size, mask.shape[2]))

        if self.use_offset:
            offset_w = (self.img_size - resized_width) // 2
            offset_h = (self.img_size - resized_height) // 2
            new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image
            new_mask[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = mask

        else:
            new_image[0:resized_height, 0:resized_width] = image
            new_mask[0:resized_height, 0:resized_width] = mask

        sample = {'img': torch.from_numpy(new_image).to(torch.float32), 'mask': torch.from_numpy(new_mask).to(torch.long), 'prompt': prompt}
        return sample


class Resizer_cv2(object):
    def __init__(self, img_size=512):
        self.img_size = img_size
    def __call__(self, sample):
        image, mask, prompt = sample['img'], sample['mask'], sample['prompt']
        new_image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        new_mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        sample = {'img': torch.from_numpy(new_image).to(torch.float32), 'mask': torch.from_numpy(new_mask).to(torch.float32), 'prompt': prompt}
        return sample


class Custom_Augment(object):
    def __init__(self,
                 gray_scale=0.5,
                 rotate=0.5,
                 scale=0.5,
                 scale_range=[1.1, 1.7, 0.2],
                 flipx=0.5,
                 flipy=0.5,
                 brightness=0.5,
                 colorjitter=0.5,
                 blur=0.5,
                 eqHis=0.5,
                 rd_crop=0.5
                 # enhance_contrast=0.5,
                 ):
        self.gray_scale = gray_scale
        self.rotate = rotate
        self.scale = scale
        self.scale_range = scale_range
        self.flipx = flipx
        self.flipy = flipy
        self.brightness = brightness
        self.colorjitter = colorjitter
        self.blur = blur
        self.eqHis = eqHis
        self.rd_crop = rd_crop

    def __call__(self, sample):
        image, mask, prompt = sample['img'], sample['mask'], sample['prompt']
        if self.gray_scale > np.random.rand():
            sample = gray_image(image, mask)
        if self.rotate > np.random.rand():
            sample = rotate_image(image, mask)
        if self.scale > np.random.rand():
            sample = scale_image(image, mask, self.scale_range)
        if self.flipx > np.random.rand():
            sample = flip_horizontal(image, mask)
        if self.flipy > np.random.rand():
            sample = flip_vertical(image, mask)
        if self.brightness > np.random.rand():
            sample = adjust_brightness(image, mask)
        if self.colorjitter > np.random.rand():
            sample = color_jitter(image, mask)
        if self.blur > np.random.rand():
            sample = blur(image, mask)
        if self.eqHis > np.random.rand():
            sample = equalize_Histogram(image, mask)
        if self.rd_crop > np.random.rand():
            sample = random_crop(image, mask)
        # if self.enhance_contrast < np.random.rand():
        #     image, mask = enhace_duong(image, mask)
        t_image, t_mask = sample['img'], sample['mask']
        sample = {'img': t_image, 'mask': t_mask, 'prompt': prompt}

        return sample


def gray_image(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    sample = {'img': gray_image, 'mask': mask}
    return sample

def rotate_image(image, mask):
    degree = np.random.randint(-80, 80)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    rotated_mask = cv2.warpAffine(mask, M, (w, h))
    sample = {'img': rotated_img, 'mask': rotated_mask}
    return sample

def scale_image(image, mask, scale=[0.5, 1.5, 0.2]):
    min, max, step = scale[0], scale[1], scale[2]
    scale = np.random.choice(np.arange(min, max, step))
    h, w = image.shape[:2]
    nw = int(w*scale)
    nh = int(h*scale)
    dim = (nw, nh)
    new_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    new_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_LINEAR)

    # offset_w = (nw - w) // 2
    # offset_h = (nh - h) // 2
    # crop_image = new_image[offset_h:offset_h + nh, offset_w:offset_w + nw]
    # crop_mask = new_mask[offset_h:offset_h + nh, offset_w:offset_w + nw]
    if scale >=1:
        cx1 = int((nw / 2) - (w / 2))
        cy1 = int((nh / 2) - (h / 2))
        cx2 = int((nw / 2) + (w / 2))
        cy2 = int((nh / 2) + (h / 2))
        scaled_image = new_image[cy1:cy2, cx1:cx2]
        scaled_mask = new_mask[cy1:cy2, cx1:cx2]
    else:
        cx1 = int((w / 2) - (nw / 2))
        cy1 = int((h / 2) - (nh / 2))
        cx2 = int((w / 2) + (nw / 2))
        cy2 = int((h / 2) + (nh / 2))
        scaled_image = np.zeros_like(image)
        scaled_mask = np.zeros_like(mask)
        scaled_image[cy1:cy2, cx1:cx2] = new_image
        scaled_mask[cy1:cy2, cx1:cx2] = new_mask

    sample = {'img': scaled_image, 'mask': scaled_mask}
    return sample

def flip_horizontal(image, mask):
    flip_image = cv2.flip(image, 1)
    flip_mask = cv2.flip(mask, 1)
    sample = {'img': flip_image, 'mask': flip_mask}
    return sample

def flip_vertical(image, mask):
    flip_image = cv2.flip(image, 0)
    flip_mask = cv2.flip(mask, 0)
    sample = {'img': flip_image, 'mask': flip_mask}
    return sample

def blur(image, mask):
    k_size = np.random.choice(np.arange(3, 9, 2))
    blur_image = cv2.blur(image, (k_size, k_size))
    sample = {'img': blur_image, 'mask': mask}
    return sample

def color_jitter(image, mask):
    bright_image = adjust_brightness(image, return_image=True)
    hsv_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2HSV)
    adj_image = adjust_contrast(hsv_image)
    adj_image = adjust_saturation(adj_image)
    adj_image = adjust_hue(adj_image)

    jitter_image = cv2.cvtColor(adj_image, cv2.COLOR_HSV2BGR)

    sample = {'img': jitter_image, 'mask': mask}
    return sample

def adjust_hue(hsv_image, factor=[0.5, 1.5]):
    hue_factor = np.random.uniform(factor[0], factor[1])
    h, s, v = cv2.split(hsv_image)
    np_h = np.array(h, dtype=np.uint8)
    np_h += np.uint8(hue_factor * 255)

    adj_image = cv2.merge([np_h, s, v])
    # adj_image = cv2.cvtColor(adj_image, cv2.COLOR_HSV2BGR)
    return adj_image

def adjust_saturation(hsv_image, factor=[0.5, 1.5]):
    s_factor = np.random.uniform(factor[0], factor[1])
    h, s, v = cv2.split(hsv_image)
    np_s = np.array(s, dtype=np.uint8)
    np_s += np.uint8(s_factor * 255)

    adj_image = cv2.merge([h, np_s, v])
    # adj_image = cv2.cvtColor(adj_image, cv2.COLOR_HSV2BGR)
    return adj_image

def adjust_contrast(hsv_image, factor=[0.5, 1.5]):
    v_factor = np.random.uniform(factor[0], factor[1])
    h, s, v = cv2.split(hsv_image)
    np_v = np.array(v, dtype=np.uint8)
    np_v += np.uint8(v_factor * 255)

    adj_image = cv2.merge([h, s, np_v])
    # adj_image = cv2.cvtColor(adj_image, cv2.COLOR_HSV2BGR)
    return adj_image

def adjust_brightness(image, mask=None, factor=[0.5, 1.5], return_image=False):
    b_factor = np.random.uniform(factor[0], factor[1])
    table = np.array([i * b_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
    adj_image = cv2.LUT(image, table)
    if return_image:
        return adj_image
    else:
        sample = {'img': adj_image, 'mask': mask}
        return sample

def equalize_Histogram(image, mask):
    if len(image.shape)==2:
        equalized = cv2.equalizeHist(image)
    else:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            equalized = np.zeros_like(image)
            equalized[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            equalized[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    # cv2.imwrite("test2.png", equalized)
    sample = {'img': equalized, 'mask': mask}
    return sample

def random_crop(image, mask):
    #save_augment(image, boxes, f_name='crop_bf')
    h, w = image.shape[0], image.shape[1]
    h_ratio = np.random.choice(np.arange(0.6, 0.9, 0.1))
    w_ratio = np.random.choice(np.arange(0.6, 0.9, 0.1))
    height = int(h*h_ratio)
    width = int(w*w_ratio)
    x = np.random.randint(0, image.shape[1] - width)
    y = np.random.randint(0, image.shape[0] - height)
    crop_img = image[y:y+height, x:x+width]
    crop_mask = mask[y:y+height, x:x+width]

    #save_augment(crop_img, boxes, f_name='crop')
    sample = {'img': crop_img, 'mask': crop_mask}
    return sample
