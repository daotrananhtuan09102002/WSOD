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

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float32)


def get_prediction(batch_cam, cam_threshold, image_size=(224, 224), gaussian_ksize=1):
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

            if cam_threshold is None:
                cam = cv2.normalize(cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                cam = cv2.GaussianBlur(cam, (gaussian_ksize, gaussian_ksize), 0)

                _, thr_gray_heatmap = cv2.threshold(
                    src=cam,
                    thresh=0,
                    maxval=255,
                    type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                cam = cv2.normalize(cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                _, thr_gray_heatmap = cv2.threshold(
                    src=cam,
                    thresh=int(cam_threshold * cam.max()),
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

        if len(prediction) == 0:
            prediction.append([-1, -1, -1, -1, -1])

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

def plot_localization_report(precision, recall, f1, num_cam_thresholds, path=None, show=False):
    """
    Args:
        precision: Array(3, num cam thresholds, num classes)
        recall: Array(3, num cam thresholds, num classes)
        f1: Array(3, num cam thresholds, num classes)
        num_cam_thresholds: int,
        path: str, path to save plot
        show: bool, whether to show plot
    Returns:
        None 
    """
    if path is None and show == False:
        return

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    thresholds = np.linspace(0.0, 0.9, num_cam_thresholds)

    for iou_idx, iou in enumerate([0.3, 0.5, 0.7]):
        avg_precision = np.mean(precision[iou_idx], axis=1)
        avg_recall = np.mean(recall[iou_idx], axis=1)
        avg_f1_score = np.mean(f1[iou_idx], axis=1)

        pos = (iou_idx//2, iou_idx%2)
        axs[pos].plot(thresholds, avg_precision, label='Avg Precision')
        axs[pos].plot(thresholds, avg_recall, label='Avg Recall')
        axs[pos].plot(thresholds, avg_f1_score, label='Avg F1-score')
        axs[pos].set_xlabel('Threshold')
        axs[pos].set_ylabel('Score')
        axs[pos].set_title(f'Score vs Threshold @ IoU={iou}')
        axs[pos].legend()
        axs[pos].grid(True)

    axs[1, 1].plot(thresholds, f1.mean(2).mean(0), label='F1 Score')
    axs[1, 1].plot(thresholds, precision.mean(2).mean(0), label='Precision')
    axs[1, 1].plot(thresholds, recall.mean(2).mean(0), label='Recall')
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].set_title('Score vs Threshold')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    
    if path is not None:
        plt.savefig(path + '/additional_info.png')
    
    if show:
        plt.show()
    
    plt.close()

def custom_report(precision, recall, f1, class_names, num_cam_thresholds, digits=4):
    """
    Args:
        precision: Array(3, num cam thresholds, num classes)
        recall: Array(3, num cam thresholds, num classes)
        f1: Array(3, num cam thresholds, num classes)
        class_names: list of class names
        num_cam_thresholds: int
    Returns:
        None 
    """

    cam_thresholds = np.linspace(0.0, 0.9, num_cam_thresholds)

    class_width = max(len(cn) for cn in class_names)
    main_header_line = f"{'':>{class_width}}|{'Average Precision':^43}|{'Average Recall':^43}|{'Average F1 Score':^43}|{'Best F1 @ CAM Thresholds':^62}|"
    print(main_header_line)

    header_line = f"{'':^{class_width}}|" + f"{'IoU30':^10s}|{'IoU50':^10s}|{'IoU70':^10s}|{'Mean':^10s}|" * 3 + f"{'IoU30':^20}|{'IoU50':^20}|{'IoU70':^20}|"
    print(header_line)
    print('-' * len(header_line))

    precision_mean_across_cam_thresholds = precision.mean(1)
    recall_mean_across_cam_thresholds = recall.mean(1)
    f1_mean_across_cam_thresholds = f1.mean(1)

    best_cam_thresholds = f1.argmax(1)

    for class_idx, class_name in enumerate(class_names):
        class_line = f'{class_name:>{class_width}}|'

        for iou_idx in range(3):
            class_line += f'{precision_mean_across_cam_thresholds[iou_idx, class_idx]:^10.{digits}f}|'
        class_line += f'{precision_mean_across_cam_thresholds[:, class_idx].mean():^10.{digits}f}|'

        for iou_idx in range(3):
            class_line += f'{recall_mean_across_cam_thresholds[iou_idx, class_idx]:^10.{digits}f}|'
        class_line += f'{recall_mean_across_cam_thresholds[:, class_idx].mean():^10.{digits}f}|'

        for iou_idx in range(3):
            class_line += f'{f1_mean_across_cam_thresholds[iou_idx, class_idx]:^10.{digits}f}|'
        class_line += f'{f1_mean_across_cam_thresholds[:, class_idx].mean():^10.{digits}f}|'

        # print f1 max with cam threshold for each iou
        for iou_idx in range(3):
            max_f1 = f'{f1[iou_idx, best_cam_thresholds[iou_idx, class_idx], class_idx]:^10.{digits}f}'
            at_cam_threshold = f'@ {cam_thresholds[best_cam_thresholds[iou_idx, class_idx]]:^8g}'
            class_line += f'{max_f1 + at_cam_threshold}|'
        print(class_line)

    print('-' * len(header_line))

    # print mean of each column
    mean_line = f'{"Mean":>{class_width}}|'
    for iou_idx in range(3):
        mean_line += f'{precision_mean_across_cam_thresholds[iou_idx].mean():^10.{digits}f}|'
    mean_line += f'{precision_mean_across_cam_thresholds.mean():^10.{digits}f}|'

    for iou_idx in range(3):
        mean_line += f'{recall_mean_across_cam_thresholds[iou_idx].mean():^10.{digits}f}|'
    mean_line += f'{recall_mean_across_cam_thresholds.mean():^10.{digits}f}|'

    for iou_idx in range(3):
        mean_line += f'{f1_mean_across_cam_thresholds[iou_idx].mean():^10.{digits}f}|'
    mean_line += f'{f1_mean_across_cam_thresholds.mean():^10.{digits}f}|'

    print(mean_line)

    # print mean f1 score across iou and cam thresholds
    print(f"Mean Average F1 Score: {f1.mean():.{digits}f}")

def process_batch(preds, cm_list, x, y, cam_threshold_idx=None):
    """ Process batch of predictions

    Args:
        preds: Array[B, None, 5], x1, y1, x2, y2, class
        cm_list: list of confusion matrix at different iou thresholds
        x: Array[B, 3, H, W], images
        y: dict of Array[B, num_classes], labels
    """
    for img_idx in range(x.shape[0]):
        for gt_class in torch.nonzero(y['labels'][img_idx]).flatten():
            pred = preds[img_idx][preds[img_idx][:, 4] == gt_class]
            gt = y['bounding_boxes'][img_idx][y['bounding_boxes'][img_idx][:, 0] == gt_class]

            npred = pred.shape[0]

            # model has no predictions on this image
            if npred == 0:
                if cam_threshold_idx is not None:
                    for cm in cm_list:
                        cm[cam_threshold_idx].process_batch(detections=None, labels=gt)
                else:
                    for cm in cm_list:
                        cm[0].process_batch(detections=None, labels=gt)
            else:
                if cam_threshold_idx is not None:
                    for cm in cm_list:
                        cm[cam_threshold_idx].process_batch(detections=pred, labels=gt)
                else:
                    for cm in cm_list:
                        cm[0].process_batch(detections=pred, labels=gt)