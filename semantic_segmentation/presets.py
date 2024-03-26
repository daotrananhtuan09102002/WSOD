# from https://github.com/pytorch/vision/blob/main/references/segmentation/presets.py
import torch
import transforms as T

class SegmentationPresetEval:
    def __init__(
        self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ):

        transforms = [
            T.Resize(base_size, base_size),
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)