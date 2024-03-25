import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader
from torcheval import metrics
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from pathlib import Path

from data_loaders import _IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE, VOCDataset, collate_fn
from wsod.resnet import resnet50
from wsod.util import get_prediction
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

def process_prediction(preds):
    return [
        {
            'boxes': image_pred[:, :4],
            'scores': image_pred[:, 5],
            'labels': image_pred[:, 4].int()
        } if (image_pred != -1).any() else
        {
            'boxes': torch.tensor([]),
            'scores': torch.tensor([]),
            'labels': torch.tensor([])
        }
        for image_pred in preds
    ]

def process_target(y):
    return [
        {
            'boxes': image_gt[:, 1:],
            'labels': image_gt[:, 0].int(),
        }
        for image_gt in y['bounding_boxes']
    ]

def get_value_from_tensor(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        else:
            return value.numpy()
    else:
        return value
    
def to_csv(args, result):
    args_dict = vars(args)
    
    for k in ['dataset_name', 'batch_size', 'data_root', 'log_dir', 'class_metrics', 'drop_prob', 'drop_threshold', 'pretrained']:
        args_dict.pop(k)

    args_dict['checkpoint_path'] = Path(args_dict['checkpoint_path']).stem
    
    res = {**args_dict, **result}

    (pd.DataFrame.from_dict(res, orient='index')
                .T
                .map(get_value_from_tensor)
                .to_csv(
                    'temp.csv',
                    index=False, 
                    header=not os.path.exists('temp.csv'), 
                    mode='a'
                )
    )

@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()
    
    num_cam_thresholds = 1 if args.use_otsu else args.num_cam_thresholds


    if args.classification_metric == 'mAP':
        cls_metric = metrics.MultilabelAUPRC(num_labels=_NUM_CLASSES_MAPPING[args.dataset_name])
    elif args.classification_metric == 'acc':
        cls_metric = metrics.MultilabelAccuracy()

    
    loc_metric = [
        MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=args.iou_thresholds,
            class_metrics=args.class_metrics
        ) for _ in range(num_cam_thresholds)
    ]

    str_cam = 'ccams' if args.no_ccam > 0 else 'cams'

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        x = x.cuda()

        if args.no_ccam > 0:
            y_pred = model(x, return_cam=True, no_ccam=args.no_ccam)    
        else:
            y_pred = model(x, return_cam=True)

        # Eval classification
        cls_metric.update(y_pred['probs'], y['labels'])

        # Eval localization
        target = process_target(y)

        if args.use_otsu:
            preds = get_prediction(
                batch_cam=y_pred[str_cam], 
                probs=y_pred['probs'],
                cam_threshold=None, 
                image_size=(args.resize_size, args.resize_size), 
                gaussian_ksize=args.gaussian_ksize
            )
            preds = process_prediction(preds)
            
            loc_metric[0].update(preds, target)
        else:
            for cam_threshold_idx, cam_threshold in enumerate(np.linspace(0.0, 0.9, num_cam_thresholds)):
                preds = get_prediction(
                    batch_cam=y_pred[str_cam],
                    probs=y_pred['probs'], 
                    cam_threshold=cam_threshold, 
                    image_size=(args.resize_size, args.resize_size)
                )
                preds = process_prediction(preds)

                loc_metric[cam_threshold_idx].update(preds, target)

    # Classification result
    cls_result = cls_metric.compute().item()

    # Localization result
    loc_result = [m.compute() for m in loc_metric]
    best_performing_cam_threshold_idx = np.argmax([r['map'].item() for r in loc_result])

    return {
        f'cls_{args.classification_metric}': cls_result,
        **loc_result[best_performing_cam_threshold_idx]
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
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    parser.add_argument('--num_cam_thresholds', type=int, default=10, help='Number of cam thresholds')
    
    # Method arguments
    parser.add_argument('--use_otsu', action='store_true', help='Use Otsu thresholding to get bounding box')
    parser.add_argument('--gaussian_ksize', type=int, default=1, help='Gaussian kernel size for gaussian blur before Otsu thresholding')
    parser.add_argument('--iou_thresholds', nargs='+', default=[0.3, 0.5, 0.7], help='IoU threshold')
    parser.add_argument('--classification_metric', type=str, default='acc', choices=['acc', 'mAP'], help='Type of classification metric')
    parser.add_argument('--class_metrics', action='store_true', help='Option to enable per-class metrics for mAP and mAR_100 for torchmetrics.detection.MeanAveragePrecision')
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
    print(result)

    # Save result to csv
    to_csv(args, result)
    
if __name__ == '__main__':
    main()