import argparse
import os
import torch
import torchvision as tv

from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils.transforms import Transform4EachKey
from at_learner_core.utils import transforms


main_transforms = [
    transforms.CropFaceParts(image_col = 'data'),
    Transform4EachKey([
        tv.transforms.Resize((240, 240)),
        tv.transforms.CenterCrop((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        key_list=['eyes', 'chin', 'nose', 'ear_l', 'ear_r']),
    Transform4EachKey([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], key_list=['data']),
]


def get_config(test_name, use_flip=False):
    additional_transforms = []
    if use_flip:
        additional_transforms = [Transform4EachKey([tv.transforms.RandomHorizontalFlip(p=1)], key_list=['data'])]

    test_transforms = tv.transforms.Compose(additional_transforms + main_transforms)

    test_config = {
        'test_config_name': test_name,
        'out_path': None,
        'ngpu': 1,
        'dataset_configs': {
            'dataset_name': 'Df2DictDataset',
            'datalist_path': '../test_out.csv',
            'data_columns': [('opensource_crop_0.3_path_black', 'data')],
            'target_columns': [('label','liveness')],
            'transform_source': 'this_config',
            'transforms': test_transforms,
            'test_process_config': {
                'metric': {
                    'name': 'acer',
                    'target_column': 'liveness',
                    'output_column': 'liveness_output'
                }
            },
            'nthreads': 8,
            'batch_size': 64,
        },

        'logger_config': {
            'logger_type': 'log_combiner',
            'loggers': [
                {'logger_type': 'test_filelogger',
                 'show_metrics': {
                     'name': 'acer',
                 },
                 'other_info': ['parts_output']
                 }]
        }
    }

    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, test_config)
    return ns_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath',
                        type=str,
                        default='experiment_tests/',
                        help='Path to save options')
    args = parser.parse_args()

    configs = get_config(test_name='test', use_flip=False)
    out_path = os.path.join(args.savepath,
                            configs.test_config_name)
    os.makedirs(out_path, exist_ok=True)
    if configs.out_path is None:
        configs.out_path = out_path
    filename = os.path.join(out_path,
                            configs.test_config_name + '.config')

    torch.save(configs, filename)

    configs = get_config(test_name='test_flip', use_flip=True)
    out_path = os.path.join(args.savepath,
                            configs.test_config_name)
    os.makedirs(out_path, exist_ok=True)
    if configs.out_path is None:
        configs.out_path = out_path
    filename = os.path.join(out_path,
                            configs.test_config_name + '.config')

    torch.save(configs, filename)
    print('Options file was saved to ' + filename)
