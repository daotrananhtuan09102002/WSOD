import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from data_loaders import _IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE, VOCDataset, collate_fn
from torch.utils.data import DataLoader
from torcheval import metrics
from tqdm import tqdm

from wsod.metrics import ConfusionMatrix
from wsod.resnet import resnet50
from wsod.util import (
    custom_report,
    get_prediction,
    plot_localization_report,
)
from wsod.vgg import vgg16

BUILTIN_MODELS = {
    'resnet50': resnet50,
    'vgg16': vgg16
}

_NUM_CLASSES_MAPPING = {
    "CUB": 200,
    "ILSVRC": 1000,
    "VOC": 20
}

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_metrics(metrics):
    maxlen = max([len(key) for key in metrics.keys()])
    print("Metrics:")
    print("" + "-" * (maxlen + 1))
    for k, v in metrics.items():
        print(f"{k.ljust(maxlen+1)}: {v:0.4f}")

def get_model(args):
    num_classes = _NUM_CLASSES_MAPPING[args.dataset_name]
    print("Loading model {}".format(args.architecture))

    return BUILTIN_MODELS[args.architecture](
            dataset_name=args.dataset_name,
            architecture_type=args.architecture_type,
            pretrained=args.pretrained,
            num_classes=num_classes,
            large_feature_map=args.large_feature_map,
            drop_threshold=args.drop_threshold,
            drop_prob=args.drop_prob
        ).cuda()

def process_batch(preds, cm_list, x, y, cam_threshold_idx=None):
    """ Process batch of predictions

    Args:
        preds: Array[B, None, 5], x1, y1, x2, y2, class
        cm_list: list of confusion matrix at different iou thresholds
        x: Array[B, 3, H, W], images
        y: dict of Array[B, num_classes], labels
    """
    for img_idx in range(x.shape[0]):
        pred = preds[img_idx][preds[img_idx][:, 0] != -1]
        gt = y['bounding_boxes'][img_idx]

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

@torch.no_grad()
def evaluate(model, dataloader, args):
    from data_loaders import VOC_CLASSES

    model.eval()
    
    num_cam_thresholds = 1 if args.use_otsu else args.num_cam_thresholds
    cm_list = [
        [ConfusionMatrix(nc=_NUM_CLASSES_MAPPING[args.dataset_name], iou_thres=iou_thres) for i in range(num_cam_thresholds)]
        for iou_thres in args.iou_thresholds
    ]

    if args.classification_metric == 'mAP':
        metric = metrics.MultilabelAUPRC(num_labels=_NUM_CLASSES_MAPPING[args.dataset_name])
    elif args.classification_metric == 'acc':
        metric = metrics.MultilabelAccuracy()

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        x = x.cuda()

        if args.localization_metric == 'gt':
            labels = y['labels']
        else: # args.localization_metric == 'pred'
            labels = None

        str_cam = 'ccams' if args.no_ccam > 0 else 'cams'

        if args.no_ccam > 0:
            y_pred = model(x, return_cam=True, labels=labels, no_ccam=args.no_ccam)    
        else:
            y_pred = model(x, return_cam=True, labels=labels)

        # Eval classification
        metric.update(y_pred['probs'], y['labels'])

        # Eval localization
        if args.use_otsu:
            preds = get_prediction(
                batch_cam=y_pred[str_cam],
                probs=y_pred['probs'], 
                cam_threshold=None, 
                image_size=(args.resize_size, args.resize_size),
                gaussian_ksize=args.gaussian_ksize
            )
            process_batch(preds, cm_list, x, y, None)
        else:
            for cam_threshold_idx, cam_threshold in enumerate(np.linspace(0.0, 0.9, num_cam_thresholds)):
                preds = get_prediction(
                    batch_cam=y_pred[str_cam], 
                    probs=y_pred['probs'],
                    cam_threshold=cam_threshold, 
                    image_size=(args.resize_size, args.resize_size)
                )
                process_batch(preds, cm_list, x, y, cam_threshold_idx)

    # Classification result
    result = metric.compute().item()
    metric.reset()

    # Localization result
    tp, fp, fn = list(zip(*[list(zip(*[(cm.tp_fp_fn()) for cm in cm_at_iou])) for cm_at_iou in cm_list]))
    tp = np.array(tp)
    fp = np.array(fp)
    fn = np.array(fn)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp+fp!=0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp+fn!=0)
    f1 = np.divide(2 * (precision * recall), precision + recall, out=np.zeros_like(tp), where=precision+recall!=0)

    if args.print_report:
        custom_report(precision, recall, f1, VOC_CLASSES, num_cam_thresholds)
    
    if not args.use_otsu and (args.plot_info or args.additional_info_path is not None):
        plot_localization_report(precision, recall, f1, num_cam_thresholds, args.additional_info_path, args.plot_info)

    return {
        args.classification_metric + f'_{args.split}': result,
        'f1' + f'_{args.split}': f1.mean(2).max(1).mean(),
        # **{
        #     f'average_f1@{iou_thres}_{args.split}': f1.mean((1, 2))[iou_thres_idx] 
        #    for iou_thres_idx, iou_thres in enumerate(args.iou_thresholds)
        # },
        # 'mean_average_precision' + f'_{args.split}': precision.mean(),
        # **{
        #     f'average_precision@{iou_thres}_{args.split}': precision.mean((1, 2))[iou_thres_idx] 
        #    for iou_thres_idx, iou_thres in enumerate(args.iou_thresholds)
        # },
        # 'mean_average_recall' + f'_{args.split}': recall.mean(),
        # **{
        #     f'average_recall@{iou_thres}_{args.split}': recall.mean((1, 2))[iou_thres_idx] 
        #    for iou_thres_idx, iou_thres in enumerate(args.iou_thresholds)
        # },
    }

def main():
    parser = argparse.ArgumentParser(description='Your script description here.')

    # Data loader arguments
    parser.add_argument('--dataset_name', type=str, default='VOC', help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./voc', help='Data root path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--resize_size', type=int, default=224, help='Resize size')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='Split to evaluate')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize images using ImageNet mean and std')
    parser.add_argument('--year', type=str, default='2007', choices=['2007', '2012'], help='VOC dataset year')

    # Misc arguments
    parser.add_argument('--checkpoint_path', required=True, type=str, default=None, help='Checkpoint path')
    parser.add_argument('--num_cam_thresholds', type=int, default=10, help='Number of cam thresholds')
    parser.add_argument('--print_report', action='store_true', help='Print localization report per class')
    parser.add_argument('--additional_info_path', type=str, default=None, help='Path to save additional info plot')
    parser.add_argument('--plot_info', action='store_true', help='Plot additional info')
    
    # Method arguments
    parser.add_argument('--use_otsu', action='store_true', help='Use Otsu thresholding to get bounding box')
    parser.add_argument('--gaussian_ksize', type=int, default=1, help='Gaussian kernel size for gaussian blur before Otsu thresholding')
    parser.add_argument('--iou_thresholds', nargs='+', default=[0.3, 0.5, 0.7], help='IoU threshold')
    parser.add_argument('--classification_metric', type=str, default='acc', choices=['acc', 'mAP'], help='Type of classification metric')
    parser.add_argument('--localization_metric', type=str, default='pred', choices=['pred', 'gt'], help='Whether to use prediction or ground truth labels for localization metric')
    parser.add_argument('--no_ccam', type=int, default=0, help='Number of CCAMs to use')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--architecture_type', type=str, default='cam', help='Model architecture type')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
    parser.add_argument('--large_feature_map', action='store_true', help='Use large feature map')
    parser.add_argument('--drop_threshold', type=float, default=0.8, help='Drop threshold')
    parser.add_argument('--drop_prob', type=float, default=0.25, help='Drop probability')

    args = parser.parse_args()

    # Use arguments in your Trainer initialization
    set_random_seed(42)

    tf = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    if args.normalize:
        tf = transforms.Compose([
            tf,
            transforms.Normalize(mean=_IMAGE_MEAN_VALUE, std=_IMAGE_STD_VALUE)
        ])

    dataloader = DataLoader(
        VOCDataset(
            root=args.data_root,
            year=args.year,
            image_set=args.split,
            transform=tf,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    model = get_model(args)

    checkpoint_path = args.checkpoint_path
    if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model = torch.nn.DataParallel(model)

    print(f"Using model:{args.architecture}-{args.architecture_type}")
    print('Evaluating model...')

    # Evaluate model
    result = evaluate(model, dataloader, args)
    print_metrics(result)
    
if __name__ == '__main__':
    main()