import argparse
import os
import torch
import torchvision as tv
from at_learner_core.utils import transforms
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils.transforms import Transform4EachKey
from shutil import copyfile



train_transforms = tv.transforms.Compose([
    tv.transforms.RandomApply([transforms.MakeTrash(data_key='data',target_key='liveness',final_label=0,trash_size=(5,10))],p=0.1),
    transforms.CropFaceParts(image_col = 'data'),
    Transform4EachKey([
        tv.transforms.Resize((336, 336)),
        transforms.AA(),
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
], key_list=['data']), 
    Transform4EachKey([
        tv.transforms.Resize((336, 336)),
        transforms.AA(),
        tv.transforms.Resize((240, 240)),
        tv.transforms.RandomCrop((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
], key_list=['eyes','chin','nose','ear_l','ear_r']),       
])


test_transforms = tv.transforms.Compose([
    transforms.CropFaceParts(image_col = 'data'),
    Transform4EachKey([
        tv.transforms.Resize((240, 240)),
        tv.transforms.CenterCrop((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])], 
        key_list=['eyes','chin','nose','ear_l','ear_r']),
    Transform4EachKey([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ], key_list=['data']),
])




def get_config():
    config = {
        'head_config': {
            'task_name': 'CASIA_Hifi',
            'exp_name': 'exp21',
            'text_comment': '',
        },

        'checkpoint_config': {
            'out_path': None,
            'save_frequency': 1,
        },

        'datalist_config': {
            'trainlist_config': {
                'dataset_name': 'Df2DictDataset',
                'datalist_path': '/netapp/grinchuk/meta/liveness/train_ssd25.txt',
                'data_columns':  [('opensource_crop_0.3_path_black', 'data')],
                'target_columns': [('label','liveness')],
                'transforms': train_transforms,
            },
            'testlist_configs': {
                'dataset_name': 'Df2DictDataset',
                'datalist_path': '/netapp/grinchuk/meta/liveness/val_ssd25.txt',
                'data_columns':  [('opensource_crop_0.3_path_black', 'data')],
                'target_columns': [('label','liveness')],
                'transforms': test_transforms,
                }
        },

        'train_process_config': {
            'nthreads': 12,
            'ngpu': 4,
            'batchsize': 48,
            'nepochs': 60,
            'resume': None,
            'optimizer_config': {
                'name': 'Adam',
                'lr_config': {
                    'lr_type': 'StepLR',
                    'lr': 0.0006,
                    'lr_decay_period': 3,
                    'lr_decay_lvl': 0.9,
                },
                'weight_decay': 0.00005,
            },
        },

        'test_process_config': {
            'metric': {
                'name': 'acer',
                'target_column': 'liveness',
                'output_column': 'liveness_output'
            }
        },

        'wrapper_config': {
            'wrapper_name': 'FPWrapper',
            'backbone': 'efficientnet-b0',
            'nclasses': [1],
            'loss': ['BCE','BCE_weighted'],
            'clf_names': ['liveness'],
            'loss_weights': [0.5,0.5,0.1,0.1,0.1,0.05,0.05],
            'loss_config': {
                'pos_weight': 5
           },
            'pretrained': None,
        },

        'logger_config': {
            'logger_type': 'log_combiner',
            'loggers': [
                {'logger_type': 'terminal',
                 'log_batch_interval': 30,
                 'show_metrics': {
                     'name': 'acer',
                 }},
                {'logger_type': 'tensorboard',
                 'log_batch_interval': 10,
                 'show_metrics': {
                     'name': 'acer',
                 }}
            ]

        },
        'manual_seed': 42,
        'resume': None,
    }

    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, config)
    return ns_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath',
                        type=str,
                        default='experiments/',
                        help='Path to save options')
    args = parser.parse_args()
    configs = get_config()
    out_path = os.path.join(args.savepath,
                            configs.head_config.task_name,
                            configs.head_config.exp_name)
    os.makedirs(out_path, exist_ok=True)
    if configs.checkpoint_config.out_path is None:
        configs.checkpoint_config.out_path = out_path
    filename = os.path.join(out_path,
                            configs.head_config.task_name + '_' + configs.head_config.exp_name + '.config')
    
    py_filename = os.path.join(out_path,
                            configs.head_config.task_name + '_' + configs.head_config.exp_name + '.py')
    
    torch.save(configs, filename)
    print('Options file was saved to ' + filename)
