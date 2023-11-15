import numpy as np
import torch
import random
from data_loaders import get_data_loader
from trainer import Trainer
from trainer import Trainer
import argparse

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='Your script description here.')

    # Data loader arguments
    parser.add_argument('--data_roots', type=str, default='./voc', help='Data roots path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--resize_size', type=int, default=224, help='Resize size')


    # Trainer arguments
    parser.add_argument('--dataset_name', type=str, default='VOC', help='Dataset name')
    parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--architecture_type', type=str, default='cam', help='Model architecture type')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
    parser.add_argument('--large_feature_map', type=bool, default=True, help='Use large feature map')
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
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    parser.add_argument('--type_metric', type=str, default='acc', help='Type metric')
    parser.add_argument('--type_loss', type=str, default='BCE', help='Type loss')
    parser.add_argument('--gamma_neg', type=int, default=4, help='Gamma negative for APL loss')
    parser.add_argument('--gamma_pos', type=int, default=0, help='Gamma positive for APL loss')
    parser.add_argument('--type_optimizer', type=str, default='SGD', help='Type optimizer')
    # Add more Trainer arguments as needed

    args = parser.parse_args()

    # Use arguments in your Trainer initialization
    set_random_seed(42)
    voc_dataloader = get_data_loader(data_roots=args.data_roots, batch_size=args.batch_size, 
                                     resize_size=args.resize_size)

    trainer = Trainer(
        args=args,
        loader=voc_dataloader,
    )
    

    for epoch in range(40):
        # Check warm epoch
        warm = True if epoch < 10 else False

        print(f'Epoch: {epoch + 1} {"(warm)" if warm else ""}')

        result = trainer.train(warm=warm)
        maxlen = max([len(key) for key in result.keys()])
        print("\tMetrics:")
        for k, v in result.items():
            print(f"\t{k.ljust(maxlen+1)}: {v:0.2f}")

        if (epoch + 1) % 2 == 0:
            result = trainer.evaluate()
            print(f'Evaluate at epoch{epoch + 1}')
            maxlen = max([len(key) for key in result.keys()])
            print("\tMetrics:")
            for k, v in result.items():
                print(f"\t{k.ljust(maxlen+1)}: {v:0.2f}")

        print("---------------------------------\n")

        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1)
        
        trainer.scheduler.step()

if __name__ == '__main__':
    main()