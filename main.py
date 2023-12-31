import numpy as np
import torch
import random
import argparse

from data_loaders import get_data_loader

from trainer import Trainer

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_metrics(metrics):
    maxlen = max([len(key) for key in metrics.keys()])
    print("\tMetrics:")
    print("\t" + "-" * (maxlen + 1))
    for k, v in metrics.items():
        print(f"\t{k.ljust(maxlen+1)}: {v:0.4f}")


def main():
    parser = argparse.ArgumentParser(description='Your script description here.')

    # Data loader arguments
    parser.add_argument('--dataset_name', type=str, default='VOC', help='Dataset name')
    parser.add_argument('--data_roots', type=str, default='./voc', help='Data roots path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--resize_size', type=int, default=224, help='Resize size')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize images using ImageNet mean and std')
    parser.add_argument('--year', type=str, default='2007', help='VOC dataset year')

    # Trainer arguments
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    parser.add_argument('--num_cam_thresholds', type=int, default=10, help='Number of cam thresholds')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every')
    parser.add_argument('--print_report', action='store_true', help='Print localization report per class')
    parser.add_argument('--additional_info_path', type=str, default=None, help='Path to save additional info plot')
    parser.add_argument('--plot_info', action='store_true', help='Plot additional info')
    parser.add_argument('--eval_classification_only', action='store_true', help='Evaluate classification only')
    
    # Method arguments
    parser.add_argument('--use_otsu', action='store_true', help='Use Otsu thresholding to get bounding box')
    parser.add_argument('--gaussian_ksize', type=int, default=1, help='Gaussian kernel size for gaussian blur before Otsu thresholding')
    parser.add_argument('--iou_thresholds', nargs='+', default=[0.3, 0.5, 0.7], help='IoU threshold')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--architecture_type', type=str, default='cam', help='Model architecture type')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
    parser.add_argument('--large_feature_map', action='store_true', help='Use large feature map')
    parser.add_argument('--drop_threshold', type=float, default=0.8, help='Drop threshold')
    parser.add_argument('--drop_prob', type=float, default=0.25, help='Drop probability')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--lr_classifier_ratio', type=float, default=10.0, help='Learning rate classifier ratio')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lr_decay_points', nargs='+', type=int, default=[21, 31], help='LR decay points')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='LR decay rate')
    parser.add_argument('--sim_fg_thres', type=float, default=0.4, help='Similarity foreground threshold')
    parser.add_argument('--sim_bg_thres', type=float, default=0.2, help='Similarity background threshold')
    parser.add_argument('--loss_ratio_drop', type=float, default=2.0, help='Loss ratio drop')
    parser.add_argument('--loss_ratio_sim', type=float, default=0.5, help='Loss ratio similarity')
    parser.add_argument('--loss_ratio_norm', type=float, default=0.05, help='Loss ratio normalization')
    parser.add_argument('--wsol_method', type=str, default='cam', help='WSOL method')
    parser.add_argument('--type_metric', type=str, default='acc', help='Type metric')
    parser.add_argument('--type_loss', type=str, default='BCE', help='Type loss')
    parser.add_argument('--gamma_neg', type=int, default=4, help='Gamma negative for APL loss')
    parser.add_argument('--gamma_pos', type=int, default=0, help='Gamma positive for APL loss')
    parser.add_argument('--type_optimizer', type=str, default='SGD', help='Type optimizer')
    parser.add_argument('--num_epoch', type=int, default=40, help='Number of epoch')
    parser.add_argument('--Taylor_expansion', action='store_true', help='Taylor expansion')
    parser.add_argument('--type_scheduler', type=str, default='MultiStepLR', help='Type scheduler')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA')
    parser.add_argument('--use_data_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--warm_epoch', type=int, default=5, help='Warm epoch')

    args = parser.parse_args()

    # Use arguments in your Trainer initialization
    set_random_seed(42)
    voc_dataloader = {
        'train': get_data_loader(
            data_roots=args.data_roots, 
            year=args.year,
            split='train',
            batch_size=args.batch_size, 
            resize_size=args.resize_size, 
            augment=args.use_data_augmentation,
            normalize=args.normalize
        ),
        'val': get_data_loader(
            data_roots=args.data_roots, 
            year=args.year,
            split='val',
            batch_size=args.batch_size, 
            resize_size=args.resize_size, 
            augment=False,
            normalize=args.normalize
        ),
    }

    trainer = Trainer(
        args=args,
        loader=voc_dataloader,
    )
    

    print(f"Using model:{args.architecture}-{args.architecture_type}")
    print("Using optimizer:", args.type_optimizer)
    print("Using scheduler:", args.type_scheduler)
    print("Using loss:", args.type_loss)
    if args.type_loss == 'APL':
        print("Using Taylor expansion:", args.Taylor_expansion)
    print("Using augmentation:", args.use_data_augmentation)
    print("Start training...")

    for epoch in range(args.num_epoch):
        # Check warm epoch
        warm = True if epoch < args.warm_epoch else False

        print(f'Epoch: {epoch + 1} {"(warm)" if warm else ""}')

        result = trainer.train(warm=warm)
        print_metrics(result)

        if (epoch + 1) % args.eval_every == 0:
            if args.eval_classification_only:
                result = trainer.evaluate_classification_only()
            else:
                result = trainer.evaluate()

            print(f'Evaluate at epoch: {epoch + 1}')
            print_metrics(result)

            if args.use_ema:
                result = trainer.evaluate_ema_model()
                print(f'Evaluate at epoch: {epoch + 1} (EMA)')
                print_metrics(result)

        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1)
        
        if args.type_scheduler == 'MultiStepLR':
            trainer.scheduler.step()

        print("---------------------------------\n")

    # Save ema model
    if args.use_ema:
        trainer.save_ema_model()

    print("Training finished!")
if __name__ == '__main__':
    main()
