import numpy as np
import torch
import random
from data_loaders import get_data_loader
from trainer import Trainer
from trainer import Trainer

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_random_seed(42)
    voc_dataloader = get_data_loader(data_roots='./voc', batch_size = 5, resize_size = 224)

    trainer = Trainer(
        dataset_name = 'VOC', 
        architecture = 'resnet50', 
        architecture_type = 'cam', 
        pretrained = True,   
        large_feature_map = True, 
        drop_threshold = 0.8, 
        drop_prob = 0.25, 
        lr = 0.002, 
        lr_classifier_ratio = 10.0,
        momentum = 0.9, 
        weight_decay = 0.0001, 
        lr_decay_points = [21, 31], 
        lr_decay_rate = 0.2,
        sim_fg_thres = 0.4, 
        sim_bg_thres = 0.2, 
        loss_ratio_drop = 2.0,
        loss_ratio_sim = 0.5, 
        loss_ratio_norm = 0.05, 
        wsol_method = 'cam', 
        loader = voc_dataloader, 
        log_dir = './result'
    )

    for epoch in range(40):
        # Check warm epoch
        warm = True if epoch < 10 else False

        print(f'Epoch: {epoch + 1} {"(warm)" if warm else ""}')

        result = trainer.train(warm=warm)
        print(result)

        if (epoch + 1) % 2 == 0:
            result = trainer.evaluate()
            print(f'Evaluate at epoch{epoch + 1}')
            print(result)

        print("---------------------------------\n")

        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1)
        
        trainer.scheduler.step()

if __name__ == '__main__':
    main()