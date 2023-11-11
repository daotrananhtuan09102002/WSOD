import numpy as np
import os
import torchvision
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataset(Dataset):
    def __init__(self, root, year, image_set, transform=None, num_classes=20):
        self.voc_data = torchvision.datasets.VOCDetection(root=root, year=year,
                                                          image_set=image_set, download=True)
        self.num_classes = num_classes
        self.transform = transform
        self.image_set = image_set

    def __len__(self):
        return len(self.voc_data)

    def __getitem__(self, index):
        image, target = self.voc_data[index]

        image = self.transform(image)

        # Extract labels from the target
        labels = target['annotation']['object']

        # Convert labels to a list of class indices, unique label
        class_indices = list(set(VOC_CLASSES.index(obj['name']) for obj in labels))

        # Convert class indices to a one-hot matrix
        label_one_hot = F.one_hot(torch.tensor(class_indices), num_classes=self.num_classes).sum(dim=0)

        return image, label_one_hot.int()

def get_data_loader(data_roots, batch_size,
                    resize_size, crop_size):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
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