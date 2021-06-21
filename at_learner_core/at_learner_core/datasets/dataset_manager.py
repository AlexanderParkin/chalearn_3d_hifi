import torch
import numpy as np
import random
import torch.utils.data
from .df2dict_dataset import Df2DictDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class DatasetManager(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_dataset(dataset_config):
        if dataset_config.dataset_name == 'Df2DictDataset':
            return Df2DictDataset(dataset_config, dataset_config.transforms)
        else:
            raise Exception('Unknown dataset type')


    @staticmethod
    def get_dataloader(dataset_config, train_process_config, shuffle=True):
        dataset = DatasetManager._get_dataset(dataset_config)

        sampler = None
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=train_process_config.batchsize,
                                                  shuffle=shuffle,
                                                  num_workers=train_process_config.nthreads,
                                                  sampler=sampler,
                                                  worker_init_fn=seed_worker)
        return data_loader

    @staticmethod
    def get_dataloader_by_args(dataset, batch_size, num_workers=8, shuffle=False, sampler=None):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  sampler=sampler,
                                                  worker_init_fn=seed_worker)
        return data_loader
