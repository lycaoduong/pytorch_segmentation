import os
import onnxruntime
import torch
import cv2
import numpy as np
from networks.u2net.u2net import U2NET, U2NETP
from networks.VitSeg_Visionin.vitseg_visionin import VitSegVisionin


def resizer(image, img_size=256, keep_ratio=True):
    if keep_ratio:
        height, width, _ = image.shape
        if height > width:
            scale = img_size / height
            resized_height = img_size
            resized_width = int(width * scale)
        else:
            scale = img_size / width
            resized_height = int(height * scale)
            resized_width = img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((img_size, img_size, image.shape[2]))

        offset_w = (img_size - resized_width) // 2
        offset_h = (img_size - resized_height) // 2
        new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image
        # new_image = torch.from_numpy(new_image).to(torch.float32)
    else:
        new_image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    return new_image

def back_sizer(image, ori_height, ori_width, keep_ratio=True):
    if keep_ratio:
        if ori_height > ori_width:
            size_back = ori_height
            image = cv2.resize(image, (size_back, size_back), interpolation=cv2.INTER_CUBIC)
            offset_w = (size_back - ori_width) // 2
            offset_h = (size_back - ori_height) // 2
        else:
            size_back = ori_width
            image = cv2.resize(image, (size_back, size_back), interpolation=cv2.INTER_CUBIC)
            offset_w = (size_back - ori_width) // 2
            offset_h = (size_back - ori_height) // 2
        new_image = image[offset_h:offset_h + ori_height, offset_w:offset_w + ori_width]
    else:
        new_image = cv2.resize(image, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
    return new_image



def normalizer(image, with_std=False, mean=[0.4264, 0.4588, 0.4730], std=[0.2093, 0.1969, 0.1993]):
    if with_std:
        mean = np.array(mean)
        std = np.array(std)
        image = ((image.astype(np.float32) / 255.0) - mean) / std
    else:
        image = image.astype(np.float32) / 255.0
    return image


class Segment_Fire_OpencvDNN(object):
    def __init__(self, onnx_dir, img_size=256, th=0.5, channel=3):
        self.model = cv2.dnn.readNetFromONNX(onnx_dir)
        self.img_size = img_size
        self.th = th
        self.ch = channel
        # self.model.setPreferableBackend()
    def get_prediction(self, image):
        height, width, _ = image.shape
        image = normalizer(image)
        image = resizer(image, img_size=self.img_size).astype(np.float32)
        # image = np.transpose(image, (2, 0, 1))
        input_blob = cv2.dnn.blobFromImage(image=image, scalefactor=1, size=(self.img_size, self.img_size), swapRB=False, crop=False, ddepth=cv2.CV_32F)
        if self.ch == 1:
            input_blob = input_blob[:, 0, :, :]
            input_blob = np.expand_dims(input_blob, axis=1)
        # print("blob: shape {}".format(input_blob.shape))
        self.model.setInput(input_blob)
        out = self.model.forward()
        out = np.squeeze(out)
        prediction = back_sizer(out, ori_height=height, ori_width=width)
        prediction = (prediction >= self.th).astype(np.uint8)
        return prediction


class HD_V2_ONNX(object):
    def __init__(self, onnx_dir, img_size=256, th=0.5, channel=3):
        if torch.cuda.is_available():
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=["CUDAExecutionProvider"])
        else:
            session = onnxruntime.InferenceSession(onnx_dir, None)
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.session = session
        self.img_size = img_size
        self.th = th
        self.ch = channel

    def get_prediction(self, layer1, layer2):
        in_tensor = np.zeros(layer1.shape[:2] + (self.ch,), dtype=float)
        in_tensor[:, :, 0] = layer1[:, :, 0]
        in_tensor[:, :, 1] = layer2[:, :, 0]
        height, width, _ = in_tensor.shape
        in_tensor = normalizer(in_tensor)
        in_tensor = resizer(in_tensor, img_size=self.img_size)
        input_blob = np.transpose(in_tensor, (2, 0, 1))
        input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32)
        preds = self.session.run([self.output_name], {self.input_name: input_blob})
        out = preds[0]
        out = np.squeeze(out)
        out = np.transpose(out, (1, 2, 0))
        prediction = back_sizer(out, ori_height=height, ori_width=width)
        prediction = (prediction >= self.th).astype(np.uint8)
        return prediction


class HD_V2_OpencvDNN(object):
    def __init__(self, onnx_dir, img_size=256, th=0.5, channel=3):
        self.model = cv2.dnn.readNetFromONNX(onnx_dir)
        self.img_size = img_size
        self.th = th
        self.ch = channel
        # self.model.setPreferableBackend()
    def get_prediction(self, layer1, layer2):
        in_tensor = np.zeros(layer1.shape[:2] + (self.ch,), dtype=float)
        in_tensor[:, :, 0] = layer1[:, :, 0]
        in_tensor[:, :, 1] = layer2[:, :, 0]
        height, width, _ = in_tensor.shape
        in_tensor = normalizer(in_tensor)
        in_tensor = resizer(in_tensor, img_size=self.img_size)
        input_blob = np.transpose(in_tensor, (2, 0, 1))
        input_blob = np.expand_dims(input_blob, axis=0)
        self.model.setInput(input_blob)
        out = self.model.forward()
        out = np.squeeze(out)
        out = np.transpose(out, (1, 2, 0))
        prediction = back_sizer(out, ori_height=height, ori_width=width)
        prediction = (prediction >= self.th).astype(np.uint8)
        return prediction


class U2Net_Prediction(object):
    def __init__(self, ckpt, img_size=512, th=0.5, channel=3, nc=2, device='cuda', opencv=False, custom=None):
        if ckpt.endswith('.pt') or ckpt.endswith('.pth'):
            if custom is None:
                model = U2NET(in_ch=channel, out_ch=nc)
            else:
                model = VitSegVisionin(backbone=custom, img_size=img_size, in_ch=channel, out_ch=nc)
            model = model.to(device)
            weight = torch.load(ckpt, map_location=device)
            model.load_state_dict(weight, strict=True)
            model.eval()
        else: # ONNX
            if opencv:
                model = cv2.dnn.readNetFromONNX(ckpt)
                if device == 'cuda':
                    cv2.cuda.setDevice(0)
                    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                if device == 'cuda':
                    providers = [
                        ('CUDAExecutionProvider', {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        })
                    ]
                    model = onnxruntime.InferenceSession(ckpt, None, providers=providers)
                else:
                    model = onnxruntime.InferenceSession(ckpt, None)
                model.get_modelmeta()
                self.input_name = model.get_inputs()[0].name
                self.output_name = model.get_outputs()[0].name
        self.model = model
        self.img_size = img_size
        self.th = th
        self.channel = channel
        self.device = device

    def onnx_export(self, save_path):
        generated_input = torch.randn((1, self.channel, self.img_size, self.img_size)).to(self.device)
        torch.onnx.export(
            self.model,
            generated_input,
            save_path,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes=None
        )
        print("Export Done")
        print("Check output: {}".format(save_path))

    def pre_processing(self, image):
        blob = image.astype(np.float32) / 255.0
        blob = cv2.resize(blob, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def post_processing(self, result, ori_height, ori_width):
        o_ori_size = cv2.resize(result, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
        return o_ori_size
    def prediction_opencv(self, image):
        height, width, _ = image.shape
        input_blob = self.pre_processing(image)
        self.model.setInput(input_blob)
        out = self.model.forward()
        out = np.squeeze(out)
        out = np.transpose(out, (1, 2, 0))
        prediction = self.post_processing(out, ori_height=height, ori_width=width)
        prediction = (prediction >= self.th).astype(np.uint8)
        return prediction

    def prediction_onnx(self, image):
        height, width, _ = image.shape
        input_blob = self.pre_processing(image)
        out = self.model.run([self.output_name], {self.input_name: input_blob})[0]
        out = np.squeeze(out)
        out = np.transpose(out, (1, 2, 0))
        prediction = self.post_processing(out, ori_height=height, ori_width=width)
        prediction = (prediction >= self.th).astype(np.uint8)
        return prediction

    def __call__(self, image):
        height, width, _ = image.shape
        input_blob = self.pre_processing(image)
        input_blob = torch.from_numpy(input_blob).to(torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(input_blob)
        out = out.cpu().numpy()
        out = np.squeeze(out, axis=0)
        out = np.transpose(out, (1, 2, 0))
        prediction = self.post_processing(out, ori_height=height, ori_width=width)
        prediction = (prediction >= self.th).astype(np.uint8)
        return prediction

class Segment_Fire(object):
    def __init__(self, model='u2net', weighted=None, device='cuda', quantized=False):
        self.device = device
        if model=='u2net':
            model = U2NET(in_ch=3, out_ch=1)
        elif model=='u2netp':
            model = U2NETP(in_ch=3, out_ch=1)
        self.model = model.to(self.device)
        if weighted is not None:
            weight = torch.load(weighted, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)
        self.model.eval()

    def get_prediction(self, image):
        height, width, _ = image.shape
        image = normalizer(image)
        image = resizer(image, img_size=256)
        image = torch.from_numpy(image).to(torch.float32)
        image = torch.unsqueeze(image, dim=0)
        image = image.permute(0, 3, 1, 2)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
        prediction = torch.squeeze(prediction)
        prediction = prediction.cpu().numpy()
        prediction = back_sizer(prediction, ori_height=height, ori_width=width)
        prediction = (prediction >= 0.5).astype(np.uint8)
        return prediction


class Segment_Human(object):
    def __init__(self, model='u2net', weighted=None, device='cuda', onnx=False, img_size=320):
        if onnx:
            self.model = cv2.dnn.readNetFromONNX(weighted)
        else:
            self.device = device
            if model=='u2net':
                model = U2NET(in_ch=3, out_ch=1)
            elif model=='u2netp':
                model = U2NETP(in_ch=3, out_ch=1)
            self.model = model.to(self.device)
            if weighted is not None:
                weight = torch.load(weighted, map_location=self.device)
                self.model.load_state_dict(weight, strict=True)
            self.model.eval()
        self.img_size = img_size
    def get_prediction(self, image, th=0.5):
        # abc
        height, width, _ = image.shape
        image = normalizer(image)
        image = resizer(image, img_size=self.img_size, keep_ratio=False)
        image = torch.from_numpy(image).to(torch.float32)
        image = torch.unsqueeze(image, dim=0)
        image = image.permute(0, 3, 1, 2)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
        prediction = torch.squeeze(prediction)
        prediction = prediction.cpu().numpy()
        prediction = back_sizer(prediction, ori_height=height, ori_width=width, keep_ratio=False)
        prediction = (prediction >= th).astype(np.uint8)
        return prediction

    def get_portrait(self, input):
        height, width, _ = input.shape
        input = resizer(input, img_size=self.img_size, keep_ratio=False)
        tmpImg = np.zeros((input.shape[0], input.shape[1], 3))
        input = input / np.max(input)

        tmpImg[:, :, 0] = (input[:, :, 2] - 0.406) / 0.225
        tmpImg[:, :, 1] = (input[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (input[:, :, 0] - 0.485) / 0.229

        # convert BGR to RGB
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = tmpImg[np.newaxis, :, :, :]
        tmpImg = torch.from_numpy(tmpImg)

        # convert numpy array to torch tensor
        image = tmpImg.type(torch.FloatTensor)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
        prediction = 1.0 - prediction[:, 0, :, :]
        prediction = self.normPRED(prediction)
        prediction = torch.squeeze(prediction)
        prediction = prediction.cpu().numpy()
        # prediction *= 255
        # prediction = prediction.astype(np.uint8)
        prediction = back_sizer(prediction, ori_height=height, ori_width=width, keep_ratio=False)
        return prediction

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

class Segment_HumanONNX(object):
    def __init__(self, onnx_dir, device='cuda', img_size=320, vram=1):
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': vram * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=providers)
        else:
            session = onnxruntime.InferenceSession(onnx_dir, None)
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.model = session
        self.img_size = img_size
    def get_prediction(self, image, th=0.5):
        # abc
        height, width, _ = image.shape
        input_blob = cv2.dnn.blobFromImage(image=image, scalefactor=1 / 255.0, size=(self.img_size, self.img_size),
                                           swapRB=False, crop=False, ddepth=cv2.CV_32F)
        prediction = self.model.run([self.output_name], {self.input_name: input_blob})[0]
        prediction = prediction[0, 0]
        prediction = back_sizer(prediction, ori_height=height, ori_width=width, keep_ratio=False)
        prediction = (prediction >= th).astype(np.uint8)
        return prediction


class NIA_Segment(object):
    def __init__(self, model_name='u2net', weighted=None, device='cuda', conf_thresh=0.5):
        self.device = device
        self.threshold = conf_thresh
        color_map = [[0, 0, 0],
                     [241, 90, 255],
                     [158, 7, 66],
                     [64, 243, 0],
                     [232, 46, 62],
                     [59, 30, 247],
                     [67, 250, 250],
                     [141, 30, 200],
                     [153, 255, 255],
                     [255, 171, 248],
                     [255, 182, 0],
                     [233, 0, 255],
                     [255, 30, 100],
                     [181, 161, 9],
                     [98, 180, 80],
                     [176, 78, 118],
                     [181, 141, 255],
                     [255, 207, 0],
                     [141, 59, 134],
                     [81, 28, 248],
                     [123, 188, 255],
                     [0, 100, 255],
                     [141, 255, 152],
                     [24, 165, 123],
                     [145, 163, 106],
                     [41, 90, 241],
                     [197, 233, 0],
                     [69, 20, 254],
                     [155, 230, 50],
                     [179, 153, 136],
                     [76, 0, 153],
                     [110, 234, 247]]

        self.color_map = np.array(color_map)
        if model_name == 'u2net':
            model = U2NETP(in_ch=3, out_ch=31)
        else:
            model = VitSegVisionin(backbone=model_name, img_size=512, in_ch=3, out_ch=31)
        self.model = model.to(self.device)
        if weighted is not None:
            weight = torch.load(weighted, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)
        self.model.eval()

    def __call__(self, in_image):
        height, width, _ = in_image.shape
        img = in_image.copy()
        img = img / 255.0
        img = cv2.resize(img, (512, 512), cv2.INTER_LINEAR)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.permute(0, 3, 1, 2).to(torch.float32)
        img = img.to(self.device)
        with torch.no_grad():
            o = self.model(img)
        o = torch.squeeze(o, dim=0)
        background = torch.zeros((1, 512, 512), device=self.device)
        o = torch.cat((background, o), dim=0)
        o = (o >= self.threshold).to(torch.int8)
        o = torch.argmax(o, dim=0)
        o = o.cpu().numpy().astype(np.uint8)
        result_img = self.color_map[o].astype(np.uint8)
        result_img = cv2.resize(result_img, (width, height), cv2.INTER_CUBIC)
        return result_img



# class Visionin_NIA(object):
#     def __init__(self, model='u2net', weighted=None, device='cuda', quantized=False):


if __name__ == '__main__':
    color_map = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255]]
    ).astype(np.uint8)
    img_dir = 'C:/Users/workspace/dataset/ha/A5_Bscan/image'
    label_dir = 'C:/Users/workspace/dataset/ha/A5_Bscan/label'
    save_dir = 'C:/Users/workspace/dataset/ha/A5_Bscan/result'
    weight_dir = 'C:/Users/workspace/dataset/ha/best_val_iou.pt'
    device = 'cuda'
    # onnx_dir = 'C:/Users/workspace/dataset/ha/sam_segmentation.onnx'
    # model = Segment_Fire_OpencvDNN(onnx_dir)
    model = Segment_Fire(model='u2netp', weighted=weight_dir, device=device)
    list_img = os.listdir(img_dir)
    for img in list_img:
        mask_f = os.path.join(label_dir, img)
        mask = cv2.imread(mask_f, 0)
        image_f = os.path.join(img_dir, img)
        image = cv2.imread(image_f)
        predict = model.get_prediction(image)
        save_img = mask + predict*2
        save_img = color_map[save_img]
        save_img = cv2.addWeighted(image, 0.7, save_img, 0.3, 0.0)
        save_f = os.path.join(save_dir, img)
        cv2.imwrite(save_f, save_img)

