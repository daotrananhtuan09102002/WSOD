import torch
import torch.nn as nn
import cv2
import numpy as np

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

        if len(prediction) == 0:
            prediction.append([-1, -1, -1, -1, -1])

        output.append(prediction)
    return torch.nested.nested_tensor(output, dtype=torch.float32)

def plot_localization_report(precision, recall, f1, num_cam_thresholds, plot_dir):
    """
    Args:
        precision: Array(3, num cam thresholds, num classes)
        recall: Array(3, num cam thresholds, num classes)
        f1: Array(3, num cam thresholds, num classes)
        num_cam_thresholds: int
    Returns:
        None 
    """

    import matplotlib.pyplot as plt

    for iou_idx, iou in enumerate([0.3, 0.5, 0.7]):
        avg_precision = np.mean(precision[iou_idx], axis=1)
        avg_recall = np.mean(recall[iou_idx], axis=1)
        avg_f1_score = np.mean(f1[iou_idx], axis=1)

        thresholds = np.linspace(0.0, 0.9, num_cam_thresholds)

        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, avg_precision, label='Average Precision')
        plt.plot(thresholds, avg_recall, label='Average Recall')
        plt.plot(thresholds, avg_f1_score, label='Average F1-score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Score vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(plot_dir + f'/iou_{iou}_score_vs_threshold.png')
        plt.close()

def custom_report(precision, recall, f1, class_names, cam_thresholds, digits=4):
    """
    Args:
        precision: Array(3, num cam thresholds, num classes)
        recall: Array(3, num cam thresholds, num classes)
        f1: Array(3, num cam thresholds, num classes)
        class_names: list of class names
        cam_thresholds: list of cam thresholds
    Returns:
        None 
    """

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


def get_localization_report(tp, fp, fn, class_names, plot_dir=None, digits=4):
    """
    Args:
        tp: Array(3, num cam thresholds, num classes)
        fp: Array(3, num cam thresholds, num classes)
        fn: Array(3, num cam thresholds, num classes)

    Returns:
        None 
    """
    _, num_cam_thresholds, _ = tp.shape

    # handle division by zero 
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp+fp!=0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp+fn!=0)
    f1 = np.divide(2 * (precision * recall), precision + recall, out=np.zeros_like(tp), where=precision+recall!=0)

    if plot_dir is not None:
        plot_localization_report(precision, recall, f1, num_cam_thresholds, plot_dir)

    custom_report(precision, recall, f1, class_names, np.linspace(0.0, 0.9, num_cam_thresholds), digits)
    

