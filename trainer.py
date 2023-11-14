from tqdm import tqdm
import wsod
import wsod.method
import os
import torch
import torch.nn as nn
import numpy as np
from torcheval import metrics
from APLloss import APLLoss


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "VOC": 20
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['fc.']
    }

    def __init__(self, dataset_name, architecture, architecture_type, pretrained,
                 large_feature_map, drop_threshold, drop_prob, lr, lr_classifier_ratio,
                 momentum, weight_decay, lr_decay_points, lr_decay_rate,
                 sim_fg_thres, sim_bg_thres, loss_ratio_drop, type_loss, type_metric,
                 loss_ratio_sim, loss_ratio_norm, wsol_method, loader, log_dir):
        self.dataset_name = dataset_name
        self.architecture = architecture
        self.architecture_type = architecture_type
        self.pretrained = pretrained
        self.large_feature_map = large_feature_map
        self.drop_threshold = drop_threshold
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_classifier_ratio = lr_classifier_ratio
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.sim_fg_thres = sim_fg_thres
        self.sim_bg_thres = sim_bg_thres
        self.loss_ratio_drop = loss_ratio_drop
        self.loss_ratio_sim = loss_ratio_sim
        self.loss_ratio_norm = loss_ratio_norm
        self.wsol_method = wsol_method

        self.model = self._set_model()
        self.model_multi = torch.nn.DataParallel(self.model)

        if type_loss == 'APL':
            self.criterion = APLLoss(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True).cuda()
        elif type_loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss().cuda()

        if type_metric == 'mAP':
            self.metrics = metrics.MultilabelAUPRC(num_labels=self._NUM_CLASSES_MAPPING[self.dataset_name]).to('cuda')
        elif type_metric == 'acc':
            self.metrics = metrics.MultilabelAccuracy().to('cuda')

        self.type_metric = type_metric
        self.type_loss = type_loss
        self.l1_loss = nn.L1Loss().cuda()
        self.optimizer = self._set_optimizer()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=lr_decay_points, 
            gamma=lr_decay_rate,
            verbose=True
        )

        self.loader = loader
        self.log_dir = log_dir


    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.dataset_name]
        print("Loading model {}".format(self.architecture))
        arch = self.architecture
        model = wsod.__dict__[arch](
            dataset_name=self.dataset_name,
            architecture_type=self.architecture_type,
            pretrained=self.pretrained,
            num_classes=num_classes,
            large_feature_map=self.large_feature_map,
            drop_threshold=self.drop_threshold,
            drop_prob=self.drop_prob)
        model = model.cuda()
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []
        param_features_name = []
        param_classifiers_name = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        def string_contains_any(string, substring_list):
            for substring in substring_list:
                if substring in string:
                    return True
            return False

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.architecture)):
                if self.architecture == 'vgg16':
                    param_features.append(parameter)
                    param_features_name.append(name)
                elif self.architecture == 'resnet50':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
            else:
                if self.architecture == 'vgg16':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
                elif self.architecture == 'resnet50':
                    param_features.append(parameter)
                    param_features_name.append(name)

        optimizer = torch.optim.SGD(
            [
                {
                    'params': param_features, 
                    'lr': self.lr
                },
                {
                    'params': param_classifiers,
                    'lr': self.lr * self.lr_classifier_ratio
                }
            ],
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True)
        return optimizer

    def _get_loss_alignment(self, feature, sim, target, eps=1e-15):
        def normalize_minmax(cams, eps=1e-15):
            """
            Args:
                cam: torch.Tensor(size=(B, H, W), dtype=np.float)
            Returns:
                torch.Tensor(size=(B, H, W), dtype=np.float) between 0 and 1.
                If input array is constant, a zero-array is returned.
            """

            B, _, _ = cams.shape
            min_value, _ = cams.view(B, -1).min(1)
            cams_minmax = cams - min_value.view(B, 1, 1)
            max_value, _ = cams_minmax.view(B, -1).max(1)
            cams_minmax /= max_value.view(B, 1, 1) + eps
            return cams_minmax

        feature_norm = torch.norm(feature, dim=1)
        feature_norm_minmax_flat = torch.flatten(normalize_minmax(feature_norm), start_dim=1)

        sim_flat = torch.flatten(sim, start_dim=2)
        
        # sim loss
        sim_fg = (feature_norm_minmax_flat > 0.6).float()
        sim_bg = (feature_norm_minmax_flat < 0.4).float()

        sim_fg_mean = (sim_fg[:, None] * sim_flat).sum(dim=-1) / (sim_fg.sum(dim=-1) + eps)[:, None]
        sim_bg_mean = (sim_bg[:, None] * sim_flat).sum(dim=-1) / (sim_bg.sum(dim=-1) + eps)[:, None]
        loss_sim = (sim_bg_mean - sim_fg_mean).data.clone().detach()
        loss_sim = torch.masked.masked_tensor(loss_sim, target.bool()).mean().get_data()

        # norm loss
        norm_fg = (sim_flat > 0).float()
        norm_bg = (sim_flat < 0).float()

        norm_fg_mean = (norm_fg * feature_norm_minmax_flat[:, None]).sum(dim=-1) / (norm_fg.sum(dim=-1) + eps)
        norm_bg_mean = (norm_bg * feature_norm_minmax_flat[:, None]).sum(dim=-1) / (norm_bg.sum(dim=-1) + eps)
        loss_norm = (norm_bg_mean - norm_fg_mean).data.clone().detach()
        loss_norm = torch.masked.masked_tensor(loss_norm, target.bool()).mean().get_data()


        return loss_sim, loss_norm

    def _wsol_training(self, images, target, warm=False):
        output_dict = self.model_multi(images, labels=target)
        logits = output_dict['logits']
        
        loss_classify = self.criterion(logits, target)
        if self.wsol_method == 'bridging-gap':
            loss_drop = self.l1_loss(output_dict['feature'], output_dict['feature_erased'])
            loss_sim, loss_norm = \
                self._get_loss_alignment(output_dict['feature'], output_dict['sim'], target)

            loss = loss_classify + self.loss_ratio_drop * loss_drop
            if not warm:
                loss += self.loss_ratio_sim * loss_sim + self.loss_ratio_norm * loss_norm
                
        elif self.wsol_method == 'cam':
            loss = loss_classify
        else:
            raise ValueError("wsol_method should be in ['bridging-gap', 'cam']")

        return logits, loss

    def _torch_save_model(self, filename):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(self.log_dir, filename))


    def save_checkpoint(self, epoch):
        self._torch_save_model(f'{epoch}_checkpoint.pth.tar')
        

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.log_dir, checkpoint_name)

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))


    def train(self, warm=False):
        self.model_multi.train()
        loader = self.loader['train']

        total_loss = 0.0
        num_images = 0

        for batch_idx, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            logits, loss = self._wsol_training(images, target, warm=warm)
            probs = self.model_multi.module.sigmoid(logits)
            self.metrics.update(probs, target)

            num_images += images.size(0)

            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        result = self.metrics.compute().item()
        self.metrics.reset()
        loss_average = total_loss / float(num_images)

        if self.type_metric == 'mAP':
            return dict(mAP=result, loss=loss_average)

        return dict(accuracy=result, loss=loss_average)
    
    def evaluate(self):
        self.model_multi.eval()
        loader = self.loader['val']

        for batch_idx, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            output_dict = self.model_multi(images)
            probs = output_dict['probs']

            self.metrics.update(probs, target)

        result = self.metrics.compute().item()
        self.metrics.reset()

        if self.type_metric == 'mAP':
            return dict(mAP_val=result)
        
        return dict(accuracy_val=result)
    





