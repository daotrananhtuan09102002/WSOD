import torch
import torch.nn as nn
import cv2

__all__ = ['remove_layer', 'replace_layer', 'initialize_weights']


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