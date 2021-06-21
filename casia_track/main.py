import argparse
import random
import os

import numpy as np
import torch
from configs import get_config
from casia_trainer import RGBRunner



def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)




def main():
    # Load options
    parser = argparse.ArgumentParser(description='Full frame liveness')
    parser.add_argument('--config',
                        type=str,
                        help='Path to config .config file. Leave blank if loading from configs.py')
    
    arg_conf = parser.parse_args()
    config = torch.load(arg_conf.config) if arg_conf.config else get_config()
    
    print('===Options==')
    d = vars(config)
    for k in d.keys():
        print(k, ':', d[k])

    """ Fix random seed """
    print('Setting seed to {}'.format(config.manual_seed))
    seed_all(config.manual_seed)

    
    # Create working directories
    os.makedirs(config.checkpoint_config.out_path, exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_config.out_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_config.out_path, 'log_files'), exist_ok=True)
    print('Directory {} was successfully created.'.format(config.checkpoint_config.out_path))

    # Training
    model = RGBRunner(config)
    model.train()


if __name__ == '__main__':
    main()


