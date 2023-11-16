import torch
import torch.nn as nn
import cv2
import numpy as np
from copy import deepcopy


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def replace_layer(state_dict, keyword1, keyword2):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword1 in key:
            new_key = key.replace(keyword1, keyword2)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def get_prediction(batch_cam, cam_threshold, image_size=(224, 224)):
    """
    Args:
        batch_cam: list of cam dict per image with class index as key and cam as value
        cam_theshold: value to threshold cam
    Returns:
        prediciton: Array[B, None, 5], x1, y1, x2, y2, class 
    """
    output = []

    for cam_per_image in batch_cam:
        prediction = []

        for class_index, cam in cam_per_image.items():
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, image_size, interpolation=cv2.INTER_CUBIC)

            _, thr_gray_heatmap = cv2.threshold(
                src=cam,
                thresh=cam_threshold * cam.max(),
                maxval=255,
                type=cv2.THRESH_BINARY
            )
            
            contours, _ = cv2.findContours(
                image=thr_gray_heatmap.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            height, width = cam.shape

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x0, y0, x1, y1 = x, y, x + w, y + h
                x1 = min(x1, width - 1)
                y1 = min(y1, height - 1)

                prediction.append([x0, y0, x1, y1, class_index])

        output.append(prediction)
    return torch.nested.nested_tensor(output, dtype=torch.float32)


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]