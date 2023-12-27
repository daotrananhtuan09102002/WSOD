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
                download=not(os.path.isfile(f'./voc/VOCdevkit/VOC{year}/ImageSets/Layout/{image_set}.txt'))
        ))
        self.num_classes = num_classes
        self.transform = transform
        self.image_set = image_set

    def __len__(self):
        return len(self.voc_data)

    def __getitem__(self, index):
        image, target = self.voc_data[index]

        # Convert class indices to a multi-hot matrix
        label_multi_hot = F.one_hot(
            torch.unique(target['labels']) - 1, 
            num_classes=self.num_classes
        ).sum(dim=0)

        if self.image_set == 'train':
            image = self.transform(image)

            return image, label_multi_hot.float()
        else:
            image, target = self.transform(image, target)

            bounding_boxes = torch.cat(
                (target['labels'][:, None] - 1 , target['boxes']), 
                dim=1
            )

            return image, { 
                'labels': label_multi_hot.float(),
                'bounding_boxes': bounding_boxes
            }
        
def collate_fn(batch):
    images = []
    labels = []
    bounding_boxes = []

    for item in batch:
        images.append(item[0])
        labels.append(item[1]['labels'])
        bounding_boxes.append(item[1]['bounding_boxes'])

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, {'labels': labels, 'bounding_boxes': bounding_boxes}

def get_data_loader(data_roots, year, split, batch_size, resize_size, augment=False, normalize=False):
    tf = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    if split == 'train' and augment:
        tf = transforms.Compose([
            tf,
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        ])
    
    if normalize:
        tf = transforms.Compose([
            tf,
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ])

    dataset = VOCDataset(
        root=data_roots,
        year=year,
        image_set=split,
        transform=tf
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=2,
        collate_fn=collate_fn if split != 'train' else None 
    )