import numpy as np
import os
import torchvision
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataset(Dataset):
    def __init__(self, root, year, image_set, transform=None, num_classes=20):
        self.voc_data = torchvision.datasets.wrap_dataset_for_transforms_v2(
            torchvision.datasets.VOCDetection(
                root=root, 
                year=year,
                image_set=image_set, 
                download=True
        ))
        self.num_classes = num_classes
        self.transform = transform
        self.image_set = image_set

    def __len__(self):
        return len(self.voc_data)

    def __getitem__(self, index):
        sample = self.voc_data[index]

        image, target = self.transform(sample)

        # Convert class indices to a multi-hot matrix
        label_multi_hot = F.one_hot(
            torch.unique(target['labels']) - 1, 
            num_classes=self.num_classes
        ).sum(dim=0)

        if self.image_set == 'train':
            return image, label_multi_hot.float()
        else:
            bounding_boxes = torch.cat(
                (target['labels'][:, None] - 1 , target['boxes']), 
                dim=1
            )
            return image, { 
                'labels': label_multi_hot.float(),
                'bounding_boxes': bounding_boxes
            }

def get_data_loader(data_roots, batch_size, resize_size):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=torchvision.transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE),
        ]))

    loader_dict = {
        'train': VOCDataset,
        'val': VOCDataset
    }

    loaders = {
        'train': DataLoader(
            loader_dict['train'](
                root=data_roots,
                year='2007',
                image_set='train',
                transform=dataset_transforms['train'],
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2),
        'val': DataLoader(
            loader_dict['val'](
                root=data_roots,
                year='2007',
                image_set='val',
                transform=dataset_transforms['val'],
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2)
    }
    return loaders