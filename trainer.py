import wsod
import wsod.method
import os
import torch
import torch.nn as nn
import numpy as np

from torcheval import metrics
from tqdm import tqdm
from APLloss import APLLoss
from wsod.resnet import resnet50
from wsod.util import add_weight_decay, ModelEma
from wsod.metrics import ConfusionMatrix
from wsod.util import get_prediction, get_localization_report


BUILTIN_MODELS = {
    'resnet50': resnet50,
}

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

    def __init__(self, args, loader):
        self.args = args

        self.model = self._set_model()
        self.model_multi = torch.nn.DataParallel(self.model)

        if args.use_ema:
            self.model_ema = ModelEma(self.model, decay=0.9997)

        if args.type_loss == 'APL':
            self.criterion = APLLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, 
                                     clip=0.05, eps=1e-8, 
                                     disable_torch_grad_focal_loss=True, 
                                     Taylor_expansion=args.Taylor_expansion).cuda()
        elif args.type_loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss().cuda()

        if args.type_metric == 'mAP':
            self.metrics = metrics.MultilabelAUPRC(num_labels=self._NUM_CLASSES_MAPPING[self.args.dataset_name])
        elif args.type_metric == 'acc':
            self.metrics = metrics.MultilabelAccuracy().to('cuda')

        self.type_metric = args.type_metric
        self.type_loss = args.type_loss
        self.l1_loss = nn.L1Loss().cuda()
        self.optimizer = self._set_optimizer()
        if args.type_scheduler == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=args.lr_decay_points, 
                gamma=args.lr_decay_rate,
                verbose=True
            )
        elif args.type_scheduler == 'OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=args.lr,
                steps_per_epoch=len(loader['train']),
                epochs=args.num_epoch,
                pct_start=0.2,
                verbose=False
            )
        self.loader = loader


    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        arch = self.args.architecture
        model = BUILTIN_MODELS[arch](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            drop_threshold=self.args.drop_threshold,
            drop_prob=self.args.drop_prob)
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
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture == 'vgg16':
                    param_features.append(parameter)
                    param_features_name.append(name)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
            else:
                if self.args.architecture == 'vgg16':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)
                    param_features_name.append(name)

        if self.args.type_optimizer == 'SGD': 
            optimizer = torch.optim.SGD(
                [
                    {
                        'params': param_features, 
                        'lr': self.args.lr
                    },
                    {
                        'params': param_classifiers,
                        'lr': self.args.lr * self.args.lr_classifier_ratio
                    }
                ],
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=True)
        elif self.args.type_optimizer == 'Adam':
            params = add_weight_decay(self.model, weight_decay=self.args.weight_decay)
            optimizer = torch.optim.Adam(
                params=params,
                weight_decay=0,
                lr=self.args.lr
            )
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
        loss_sim = torch.masked.masked_tensor(loss_sim, target.bool()).prod(dim=1).mean().get_data()

        # norm loss
        norm_fg = (sim_flat > 0).float()
        norm_bg = (sim_flat < 0).float()

        norm_fg_mean = (norm_fg * feature_norm_minmax_flat[:, None]).sum(dim=-1) / (norm_fg.sum(dim=-1) + eps)
        norm_bg_mean = (norm_bg * feature_norm_minmax_flat[:, None]).sum(dim=-1) / (norm_bg.sum(dim=-1) + eps)
        loss_norm = (norm_bg_mean - norm_fg_mean).data.clone().detach()
        loss_norm = torch.masked.masked_tensor(loss_norm, target.bool()).prod(dim=1).mean().get_data()


        return loss_sim, loss_norm

    def _wsol_training(self, images, target, warm=False):
        output_dict = self.model_multi(images, labels=target)
        logits = output_dict['logits']
        
        loss_classify = self.criterion(logits, target)
        if self.args.wsol_method == 'bridging-gap':
            loss_drop = self.l1_loss(output_dict['feature'], output_dict['feature_erased'])
            loss_sim, loss_norm = \
                self._get_loss_alignment(output_dict['feature'], output_dict['sim'], target)

            loss = loss_classify + self.args.loss_ratio_drop * loss_drop
            if not warm:
                loss += self.args.loss_ratio_sim * loss_sim + self.args.loss_ratio_norm * loss_norm
                
        elif self.args.wsol_method == 'cam':
            loss = loss_classify
        else:
            raise ValueError("wsol_method should be in ['bridging-gap', 'cam']")

        return logits, loss

    def _torch_save_model(self, filename):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(self.args.log_dir, filename))


    def save_checkpoint(self, epoch):
        self._torch_save_model(f'{epoch}_checkpoint.pth.tar')
        

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.args.log_dir, checkpoint_name)

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))

    def save_ema_model(self):
        checkpoint_path = os.path.join(self.args.log_dir, f'ema.pth.tar')
        torch.save({'state_dict': self.model_ema.module.state_dict()}, checkpoint_path)


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

            if self.args.type_scheduler == 'OneCycleLR':
                self.scheduler.step()
            
            if self.args.use_ema:
                self.model_ema.update(self.model)

        result = self.metrics.compute().item()
        self.metrics.reset()
        loss_average = total_loss / float(num_images)

        if self.type_metric == 'mAP':
            return dict(mAP=result, loss=loss_average)

        return dict(accuracy=result, loss=loss_average)
    
    @torch.no_grad()
    def eval_classification_only(self, split='val'):
        self.model_multi.eval()
        loader = self.loader[split]

        for batch_idx, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()

            output_dict = self.model_multi(images)
            probs = output_dict['probs']

            self.metrics.update(probs, target['labels'])

        result = self.metrics.compute().item()
        self.metrics.reset()

        if self.type_metric == 'mAP':
            return dict(mAP_val=result)
        
        return dict(accuracy_val=result)
    
    @torch.no_grad()
    def evaluate(self, split='val'):
        self.model_multi.eval()
        loader = self.loader[split]
        
        num_cam_thresholds = self.args.num_cam_thresholds
        cm_30 = [ConfusionMatrix(nc=self._NUM_CLASSES_MAPPING[self.args.dataset_name], iou_thres=0.3) for i in range(num_cam_thresholds)]
        cm_50 = [ConfusionMatrix(nc=self._NUM_CLASSES_MAPPING[self.args.dataset_name], iou_thres=0.5) for i in range(num_cam_thresholds)]
        cm_70 = [ConfusionMatrix(nc=self._NUM_CLASSES_MAPPING[self.args.dataset_name], iou_thres=0.7) for i in range(num_cam_thresholds)]

        for batch_idx, (x, y) in enumerate(tqdm(loader)):
            x = x.cuda()

            y_pred = self.model_multi(x, return_cam=True, labels=y['labels'])

            # Eval classification
            self.metrics.update(y_pred['probs'], y['labels'])

            # Eval localization
            for cam_threshold_idx, cam_threshold in enumerate(np.linspace(0.0, 0.9, num_cam_thresholds)):
                preds = get_prediction(y_pred['cams'], cam_threshold)

                for img_idx in range(x.shape[0]):
                    for gt_class in torch.unique(torch.nonzero(y['labels'][img_idx]).flatten()):
                        pred = preds[img_idx][preds[img_idx][:, 4] == gt_class]
                        gt = y['bounding_boxes'][img_idx][y['bounding_boxes'][img_idx][:, 0] == gt_class]

                        npred = pred.shape[0]

                        # model has no predictions on this image
                        if npred == 0:
                            cm_30[cam_threshold_idx].process_batch(detections=None, labels=gt)
                            cm_50[cam_threshold_idx].process_batch(detections=None, labels=gt)
                            cm_70[cam_threshold_idx].process_batch(detections=None, labels=gt)
                        else:
                            cm_30[cam_threshold_idx].process_batch(detections=pred, labels=gt)
                            cm_50[cam_threshold_idx].process_batch(detections=pred, labels=gt)
                            cm_70[cam_threshold_idx].process_batch(detections=pred, labels=gt)

        # Classification result
        result = self.metrics.compute().item()
        self.metrics.reset()

        # Localization result
        tp, fp, fn = list(zip(*[list(zip(*[(cm.tp_fp_fn()) for cm in cm_at_iou])) for cm_at_iou in [cm_30, cm_50, cm_70]]))
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp+fp!=0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp+fn!=0)
        f1 = np.divide(2 * (precision * recall), precision + recall, out=np.zeros_like(tp), where=precision+recall!=0)

        if self.args.print_report:
            from data_loaders import VOC_CLASSES
            get_localization_report(precision, recall, f1, VOC_CLASSES)

        if self.type_metric == 'mAP':
            return dict(mAP_val=result, mean_average_f1=f1.mean())
          
        return dict(accuracy_val=result, mean_average_f1=f1.mean())
        
    @torch.no_grad()
    def evaluate_ema_model(self, split='val'):
        self.model_ema.module.eval()
        loader = self.loader[split]

        for batch_idx, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()

            output_dict = self.model_ema.module(images)
            probs = output_dict['probs']

            self.metrics.update(probs, target['labels'])

        result = self.metrics.compute().item()
        self.metrics.reset()

        if self.type_metric == 'mAP':
            return dict(mAP_val_ema=result)

        return dict(accuracy_val_ema=result)