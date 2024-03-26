# from https://github.com/pytorch/vision/blob/main/references/segmentation/train.py
import sys
import os
sys.path.append('/content/WSOD')

import warnings
import random

import cv2
import torch
import torch.utils.data
import torchvision
import numpy as np

import utils
import presets

from wsod.resnet import resnet50
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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_dataset(args):
    def voc(*args, **kwargs):
        return torchvision.datasets.VOCSegmentation(
            *args, **kwargs
        )

    paths = {
        'VOC': (args.data_path, voc, 21), # 20 classes + background
    }
    p, ds_fn, num_classes = paths[args.dataset]

    
    ds = ds_fn(
        p, 
        image_set=args.image_set, 
        year=args.year, 
        transforms=get_transform(),
        download=not(os.path.isfile(f'../voc_segmentation/VOCdevkit/VOC{args.year}/ImageSets/Layout/{args.image_set}.txt'))
    )

    return ds, num_classes

def get_transform():
    return presets.SegmentationPresetEval(base_size=224)

def get_model(args):
    num_classes = _NUM_CLASSES_MAPPING[args.dataset]
    print("Loading model {}".format(args.architecture))

    return BUILTIN_MODELS[args.architecture](
            dataset_name=args.dataset,
            architecture_type=args.architecture_type,
            pretrained=args.pretrained,
            num_classes=num_classes,
            large_feature_map=args.large_feature_map,
            drop_threshold=args.drop_threshold,
            drop_prob=args.drop_prob
        )

def get_prediction(batch_cam, probs, cam_threshold, image_size=(224, 224), gaussian_ksize=1):
    """
    Args:
        batch_cam: list of cam dict per image with class index as key and cam as value
        probs: Array[B, num_classes], probability of each class
        cam_theshold: value to threshold cam
    Returns:
        prediciton: Array[B, None, 6], x1, y1, x2, y2, class, confidence(using class probability as confidence)
    """
    prediction = []

    for cam_per_image, prob in zip(batch_cam, probs):
        segmentation = np.zeros(image_size)

        sorted, indices = torch.sort(prob, descending=False)

        for class_index in indices[sorted.round().to(torch.bool)]:
            cam = cam_per_image[class_index.item()]
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
            
            segmentation[thr_gray_heatmap.astype(np.bool8)] = class_index.item() + 1 


        prediction.append(segmentation)
    return torch.tensor(np.array(prediction), dtype=torch.int64)

def get_value_from_tensor(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        else:
            return value.numpy()
    else:
        return value
    
def to_csv(args, result):
    import pandas as pd
    from pathlib import Path
    args_dict = vars(args)
    
    for k in ['data_path', 'device', 'workers', 'output_dir', 'drop_prob', 'drop_threshold', 'pretrained', 'distributed']:
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

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    
    str_cam = 'ccams' if args.no_ccam > 0 else 'cams'
    num_cam_thresholds = 1 if args.use_otsu else args.num_cam_thresholds
    
    confmat = [
        utils.ConfusionMatrix(num_classes)
        for _ in range(num_cam_thresholds)
    ]
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    num_processed_samples = 0


    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            if args.no_ccam > 0:
                output = model(image, return_cam=True, no_ccam=args.no_ccam)    
            else:
                output = model(image, return_cam=True)

            if args.use_otsu:
                preds = get_prediction(
                    batch_cam=output[str_cam], 
                    probs=output['probs'],
                    cam_threshold=None, 
                    image_size=(224, 224), 
                    gaussian_ksize=args.gaussian_ksize
                )

                preds = preds.to(device)
                confmat[0].update(target.flatten(), preds.flatten())
                
            else:
                for cam_threshold_idx, cam_threshold in enumerate(np.linspace(0.0, 0.9, num_cam_thresholds)):
                    preds = get_prediction(
                        batch_cam=output[str_cam],
                        probs=output['probs'], 
                        cam_threshold=cam_threshold, 
                        image_size=(224, 224)
                    )

                    preds = preds.to(device)
                    confmat[cam_threshold_idx].update(target.flatten(), preds.flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        for cm in confmat:
            cm. reduce_from_all_processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, '__len__')
        and len(data_loader.dataset) != num_processed_samples
    ):
        # See FIXME above
        warnings.warn(
            f'It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} '
            'samples were used for the validation, which might bias the results. '
            'Try adjusting the batch size and / or the world size. '
            'Setting the world size to 1 is always a safe bet.'
        )

    return confmat

def main(args):
    set_random_seed(42)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    model = get_model(args)
    model.to(device)

    checkpoint_path = args.checkpoint_path
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("Check {} loaded.".format(checkpoint_path))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)

    best_threshold = np.argmax([cm.compute()[2].mean().item() for cm in confmat])
    best_result = confmat[best_threshold].compute()
    result = {
        'global_correct': best_result[0].item(),
        'average_row_correct': best_result[1].tolist(),
        'IoU': best_result[2].tolist(),
        'mean_IoU': best_result[2].mean().item(),
    }

    to_csv(args, result)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='../voc_segmentation', type=str, help='dataset path')
    parser.add_argument('--dataset', default='VOC', type=str, help='dataset name')
    parser.add_argument('--image-set', default='val', type=str, choices=['train', 'val', 'test'],help='image set')
    parser.add_argument('--year', default='2007', type=str, choices=['2007', '2012'], help='dataset year')
    parser.add_argument('--device', default='cuda', type=str, help='device (Use cuda or cpu Default: cuda)')
    # parser.add_argument('-b', '--batch-size', default=8, type=int, help='batch size (default: 8)')
    parser.add_argument(
        '-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)'
    )
    parser.add_argument('--output-dir', default='.', type=str, help='path to save outputs')

    # Misc arguments
    parser.add_argument('--checkpoint-path', required=True, type=str, default=None, help='Checkpoint path')
    parser.add_argument('--num-cam-thresholds', type=int, default=10, help='Number of cam thresholds')

    # Method arguments
    parser.add_argument('--use-otsu', action='store_true', help='Use Otsu thresholding to get bounding box')
    parser.add_argument(
        '--gaussian-ksize', type=int, default=1, help='Gaussian kernel size for gaussian blur before Otsu thresholding'
    )
    parser.add_argument('--iou-thresholds', nargs='+', default=[0.3, 0.5, 0.7], help='IoU threshold')
    parser.add_argument('--no-ccam', type=int, default=0, help='Number of CCAMs to use')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--architecture-type', type=str, default='cam', help='Model architecture type')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
    parser.add_argument('--large-feature-map', action='store_true', help='Use large feature map')
    parser.add_argument('--drop-threshold', type=float, default=0.8, help='Drop threshold')
    parser.add_argument('--drop-prob', type=float, default=0.25, help='Drop probability')

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)