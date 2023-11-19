import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings

from pathlib import Path

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    # From https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    # Only remove "if n" line from the ref code
    def __init__(self, nc, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.iou_thres = iou_thres

    def save(self, save_dir='', name=''):
        if len(name) == 0:
            raise ValueError('Name of the confusion matrix must be provided.')

        np.save(Path(save_dir) / f'{name}_iou@{self.iou_thres}', self.matrix)

    def load(self, path=''):
        self.iou_thres = np.float32(Path(path).stem.split('@')[-1])
        self.matrix = np.load(path)
        self.nc = self.matrix.shape[0] - 1

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 5]), x1, y1, x2, y2, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels[:, 0].int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 4].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp_fn(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1], fn[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(15, 12), tight_layout='constrained')
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.9)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')

        if len(save_dir) == 0:
            plt.show()
        else:
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))