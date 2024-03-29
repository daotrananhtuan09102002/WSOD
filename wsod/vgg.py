"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .method.drop import AttentiveDrop
from .util import remove_layer
from .util import initialize_weights

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

configs_dict = {
    'cam': { # basic architecture
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
    'drop': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'D', 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'D', 512, 512, 512],
    }
}


class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        self.features = features

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.modules(), init_mode='he')


    @torch.no_grad()
    def compute_ccam(self, features, logits, no_ccam):
        cam_reverse = []
        cam_reverse_sum = []

        for logit, feature in zip(logits, features):
            cam_reverse_per_image = dict()
            reversed_i = torch.argsort(logit)[:no_ccam]

            for i in reversed_i:
                cam_reverse_weights = self.fc.weight[i]
                cam_reverse_per_image[i] = (cam_reverse_weights[:,None,None] * feature).mean(0, keepdim=False)

            cam_reverse_sum_per_image = torch.stack(list(cam_reverse_per_image.values())).sum(0)
            cam_reverse_sum.append(cam_reverse_sum_per_image)
            cam_reverse.append(cam_reverse_per_image)

        return cam_reverse, cam_reverse_sum


    def forward(self, x, labels=None, return_cam=False, no_ccam=None):
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if self.training:
            return {'logits': logits}

        probs = self.sigmoid(logits)

        if no_ccam is not None:
            cams = []
            cams_reverse, cams_reverse_sum = self.compute_ccam(x, logits, no_ccam)
            ccams = []
            feature_map = x.detach().clone()

            if labels is None:
                labels = torch.round(probs)
                
            # process one image at a time from batch 
            for label, feature, cam_reverse_sum in zip(labels, feature_map, cams_reverse_sum):
                cam_per_image = dict()
                ccam_per_image = dict()

                for nonzeros in label.nonzero():
                    i = nonzeros.item()
                    cam_weights = self.fc.weight[i]
                    cam_per_image[i] = (cam_weights[:,None,None] * feature).mean(0, keepdim=False)
                    ccam_per_image[i] = cam_per_image[i] - cam_reverse_sum
                    
                cams.append(cam_per_image)
                ccams.append(ccam_per_image)

            return {'probs': probs, 'cams': cams, 'cams_reverse': cams_reverse, 'ccams': ccams}
        
        if return_cam:
            cams = []
            feature_map = x.detach().clone()

            if labels is None:
                labels = torch.round(probs)

            for label, feature in zip(labels, feature_map):
                cam_per_image = dict()
                for nonzeros in label.nonzero():
                    i = nonzeros.item()
                    cam_weights = self.fc.weight[i]
                    cam_per_image[i] = (cam_weights[:,None,None] * feature).mean(0, keepdim=False)

                cams.append(cam_per_image)
            return {'probs': probs, 'cams': cams}
        
        return {'probs': probs}


class VggDrop(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggDrop, self).__init__()
        self.features = features

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.modules(), init_mode='he')

    @torch.no_grad()
    def compute_ccam(self, features, logits, no_ccam):
        cam_reverse = []
        cam_reverse_sum = []

        for logit, feature in zip(logits, features):
            cam_reverse_per_image = dict()
            reversed_i = torch.argsort(logit)[:no_ccam]

            for i in reversed_i:
                cam_reverse_weights = self.fc.weight[i]
                cam_reverse_per_image[i] = (cam_reverse_weights[:,None,None] * feature).mean(0, keepdim=False)

            cam_reverse_sum_per_image = torch.stack(list(cam_reverse_per_image.values())).sum(0)
            cam_reverse_sum.append(cam_reverse_sum_per_image)
            cam_reverse.append(cam_reverse_per_image)

        return cam_reverse, cam_reverse_sum
    
    
    def forward(self, x, labels=None, return_cam=False, no_ccam=None):
        _unerased_x = self.features[0](x)
        unerased_x = self.features[2](_unerased_x)

        unerased_x = self.conv6(unerased_x)
        unerased_x = self.relu(unerased_x)
        pre_logit_unerased = self.avgpool(unerased_x)
        pre_logit_unerased = pre_logit_unerased.view(pre_logit_unerased.size(0), -1)
        logits = self.fc(pre_logit_unerased)

        erased_x = self.features[1](_unerased_x)
        erased_x = self.features[2](erased_x)
        erased_x = self.conv6(erased_x)
        erased_x = self.relu(erased_x)

        x_normalized = F.normalize(unerased_x, dim=1)
        weight_normalized = F.normalize(self.fc.weight, dim=1)
        sim = F.conv2d(input=x_normalized, weight=weight_normalized.view(*weight_normalized.shape, 1, 1))

        if self.training:
            return {'logits': logits, 'feature': unerased_x, 'feature_erased': erased_x, 'sim': sim}

        probs = self.sigmoid(logits)

        if no_ccam is not None:
            cams = []
            cams_reverse, cams_reverse_sum = self.compute_ccam(unerased_x, logits, no_ccam)
            ccams = []
            feature_map = unerased_x.detach().clone()

            if labels is None:
                labels = torch.round(probs)
                
            # process one image at a time from batch 
            for label, feature, cam_reverse_sum in zip(labels, feature_map, cams_reverse_sum):
                cam_per_image = dict()
                ccam_per_image = dict()

                for nonzeros in label.nonzero():
                    i = nonzeros.item()
                    cam_weights = self.fc.weight[i]
                    cam_per_image[i] = (cam_weights[:,None,None] * feature).mean(0, keepdim=False)
                    ccam_per_image[i] = cam_per_image[i] - cam_reverse_sum
                    
                cams.append(cam_per_image)
                ccams.append(ccam_per_image)

            return {'probs': probs, 'cams': cams, 'cams_reverse': cams_reverse, 'ccams': ccams}

        if return_cam:
            cams = []
            feature_map = unerased_x.detach().clone()

            if labels is None:
                labels = torch.round(probs)

            for label, feature in zip(labels, feature_map):
                cam_per_image = dict()
                for nonzeros in label.nonzero():
                    i = nonzeros.item()
                    cam_weights = self.fc.weight[i]
                    cam_per_image[i] = (cam_weights[:,None,None] * feature).mean(0, keepdim=False)

                cams.append(cam_per_image)
            return {'probs': probs, 'cams': cams}
        
        return {'probs': probs}


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        multiply_hundred = False
        for key, _ in iterator:
            if key.startswith('features.'):
                if len(key.strip().split('.')) == 3:
                    keys.append(int(key.strip().split('.')[1].strip()))
                elif len(key.strip().split('.')) > 3:
                    keys.append(int(key.strip().split('.')[1].strip())*100 + int(key.strip().split('.')[2].strip()))
                    multiply_hundred = True
                else:
                    raise ValueError("Check model parameter name.")
        return sorted(list(set(keys)), reverse=True), multiply_hundred

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    def _align_keys_multiply_hundred(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2 // 100) + '.' + str(key2 % 100) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys, _ = _get_keys(pretrained_model, 'pretrained')
    current_keys, multiply_hundred = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        if multiply_hundred:
            pretrained_model = _align_keys_multiply_hundred(pretrained_model, p_key, c_key)
        else:
            pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


def load_pretrained_model(model, path=None):
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True)

    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, **kwargs):
    layer_seq = []
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'D':
            layer_seq.append(nn.Sequential(*layers))
            layer_seq.append(AttentiveDrop(kwargs['drop_threshold'], kwargs['drop_prob']))
            layers = []
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if len(layer_seq) == 0:
        return nn.Sequential(*layers)
    else:
        if len(layers) > 0:
            layer_seq.append(nn.Sequential(*layers))
        return nn.ModuleList(layer_seq)


def vgg16(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    config_key = '28x28' if kwargs['large_feature_map'] else '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)
    model = {'cam': VggCam,
             'drop': VggDrop}[architecture_type](layers, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, path=pretrained_path)
    return model