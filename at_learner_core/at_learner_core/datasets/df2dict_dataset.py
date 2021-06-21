from PIL import Image
import pandas as pd
import torch.utils.data
from collections import OrderedDict
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

def rgb_loader(path):
    return Image.open(path)#.convert('RGB')


def greyscale_loader(path):
    return Image.open(path).convert('L')


class Df2DictDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_config, transforms):
        self.datalist_config = datalist_config
        self.transforms = transforms
        self.df = self._read_list()
        if hasattr(self.datalist_config, 'dataloader') and self.datalist_config.dataloader == 'greyscale_loader':
            self.loader = greyscale_loader
        else:
            self.loader = rgb_loader

    def __getitem__(self, index):
        item_dict = OrderedDict()
        for column, column_name in self.data_columns:
            item_dict[column_name] = self.loader(self.df[column].values[index])
        if self.transforms is not None:
            item_dict = self.transforms(item_dict)

        for column, column_name in self.target_columns:
            item_dict[column_name] = torch.Tensor([self.df[column].values[index]])

        return item_dict

    def __len__(self):
        return len(self.df)

    def _read_list(self):
        data_df = pd.read_csv(self.datalist_config.datalist_path)
        if isinstance(self.datalist_config.data_columns, list):
            self.data_columns = self.datalist_config.data_columns
        elif isinstance(self.datalist_config.data_columns, tuple):
            self.data_columns = [self.datalist_config.data_columns]
        elif isinstance(self.datalist_config.data_columns, str):
            self.data_columns = [(self.datalist_config.data_columns,
                                  self.datalist_config.data_columns)]
        else:
            raise Exception('Unknown columns types in dataset')

        if isinstance(self.datalist_config.target_columns, list):
            self.target_columns = self.datalist_config.target_columns
        elif isinstance(self.datalist_config.target_columns, tuple):
            self.target_columns = [self.datalist_config.target_columns]
        elif isinstance(self.datalist_config.target_columns, str):
            self.target_columns = [(self.datalist_config.target_columns,
                                    self.datalist_config.target_columns)]
        elif self.datalist_config.target_columns is None:
            self.target_columns = []
        else:
            raise Exception('Unknown columns types in dataset')

        needed_columns = [x[0] for x in self.data_columns]
        needed_columns = needed_columns + [x[0] for x in self.target_columns]

        if hasattr(self.datalist_config, 'additional_columns_dict'):
            self.additional_columns_dict = self.datalist_config.additional_columns_dict.__dict__
            for additional_column in self.additional_columns_dict.values():
                needed_columns.append(additional_column)

        needed_columns = list(set(needed_columns))
        data_df = data_df[needed_columns]
        return data_df
