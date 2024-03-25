# from https://github.com/pytorch/vision/blob/main/references/segmentation/presets.py
import torch
import torchvision.transforms.v2 as T
import torchvision.tv_tensors
import v2_extras


class SegmentationPresetEval:
    def __init__(
        self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), backend="pil"
    ):

        transforms = [
            T.ToImage(),
            T.Resize(size=(base_size, base_size)),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)