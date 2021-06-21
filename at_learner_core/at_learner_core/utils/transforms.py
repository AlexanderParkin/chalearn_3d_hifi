import random
import collections
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import torch

import scipy.sparse
from sklearn import svm
import torchvision as tv

from torchvision.transforms import functional as F


import albumentations as albu
image_size = (336,336)


class CropFaceParts(object):
    def __init__(self, image_col='data'):
        self.global_x = [0.27,0.7379,0.5193,0.3210,0.6977]
        self.global_y = [0.4034,0.3984,0.6036,0.7952,0.7891]
        self.wh_scale_factor = 0.7765
        self.bbox_scale_factor = 1.6
        self.base_image_size = 250
        self.image_col = image_col
        
    def __call__(self, item_dict):
        image = item_dict[self.image_col]
        image = np.array(image)
        bbox = self._get_average_bbox(image)
        
        eye_crop = self._eye_crop(image,bbox)
        item_dict['eyes'] = eye_crop
        
        chin_crop = self._chin_crop(image,bbox)
        item_dict['chin'] = chin_crop
        
        nose_crop = self._nose_crop(image,bbox)
        item_dict['nose'] = nose_crop
        
        ear_l_crop,ear_r_crop = self._ears_crop(image,bbox)
        item_dict['ear_l'] = ear_l_crop
        item_dict['ear_r'] = ear_r_crop
        return item_dict

    def _get_average_bbox(self, img):
        bbox_h = img.shape[1] / self.bbox_scale_factor
        bbox_w = bbox_h * self.wh_scale_factor
        bbox_y = (img.shape[1] - bbox_h) / 2
        bbox_x = (img.shape[0] - bbox_w) / 2
        return bbox_x,bbox_y,bbox_w,bbox_h
    
    def _eye_crop(self, img, bbox):
        bbox_x,bbox_y,bbox_w,bbox_h = bbox
        xc = int((self.global_x[1] * bbox_w + self.global_x[0] * bbox_w) / 2)
        yc = int((self.global_y[1] * bbox_h + self.global_y[0] * bbox_h) / 2)
        
        k = bbox_h* 1.3 / self.base_image_size
        shift=[97,40]
        x0_ = xc - shift[0] * k
        x1_ = xc + shift[0] * k
        y0_ = yc - shift[1] * k
        y1_ = yc + shift[1] * k

        cut = img[max(0, int(bbox_y + y0_)):min(int(bbox_y + y1_), img.shape[0]),
              max(0, int(bbox_x + x0_)):min(int(bbox_x + x1_), img.shape[1])]
        if cut.shape[0] == 0 or cut.shape[1] == 0:
            cut = np.zeros((1, 1, 3)).astype(np.uint8)

        return Image.fromarray(cut)
    
    
    def _chin_crop(self, img, bbox):
        bbox_x,bbox_y,bbox_w,bbox_h = bbox
        xc = int((self.global_x[4] * bbox_w + self.global_x[3] * bbox_w) / 2)
        yc = int((self.global_y[4] * bbox_h + self.global_y[3] * bbox_h) / 2)

        k = bbox_h*1.3 / self.base_image_size
        shift=[77,20]
        x0_ = xc - shift[0] * k
        x1_ = xc + shift[0] * k
        y0_ = yc - shift[1] * k
        y1_ = yc + 3*shift[1] * k
        cut = img[max(0, int(bbox_y + y0_)):min(int(bbox_y + y1_), img.shape[0]),
              max(0, int(bbox_x + x0_)):min(int(bbox_x + x1_), img.shape[1])]
        if cut.shape[0] == 0 or cut.shape[1] == 0:
            cut = np.zeros((1, 1, 3)).astype(np.uint8)
        return Image.fromarray(cut)


    def _nose_crop(self, img, bbox):
        bbox_x,bbox_y,bbox_w,bbox_h = bbox
        xc = int(self.global_x[2] * bbox_w)
        yc = int(self.global_y[2] * bbox_h)
        k = bbox_h*1.3 / self.base_image_size
        
        shift = [40,30]
        x0_ = xc - shift[0] * k
        x1_ = xc + shift[0] * k

        y0_ = yc - shift[1] * k
        y1_ = yc + shift[1] * k
        
        cut = img[max(0, int(bbox_y + y0_)):min(int(bbox_y + y1_), img.shape[0]),
              max(0, int(bbox_x + x0_)):min(int(bbox_x + x1_), img.shape[1])]
        if cut.shape[0] == 0 or cut.shape[1] == 0:
            cut = np.zeros((1, 1, 3)).astype(np.uint8)
        return Image.fromarray(cut)


    def _ears_crop(self, img, bbox):
        bbox_x,bbox_y,bbox_w,bbox_h = bbox
        xc = int((self.global_x[1] * bbox_w + self.global_x[0] * bbox_w) / 2)
        yc = int((self.global_y[1] * bbox_h + self.global_y[0] * bbox_h) / 2)

        k = bbox_h*1.3 / self.base_image_size

        shift_l = [130,40]
        shift_r = [50,40]
        x0_ = xc - shift_l[0] * k
        x01_ = xc - shift_r[0] * k
        x10_ = xc + shift_r[0] * k
        x1_ = xc + shift_l[0] * k

        y0_ = yc - shift_r[1] * k
        y1_ = yc + shift_l[1] * k
        
        cut1 = img[max(0, int(bbox_y + y0_)):min(int(bbox_y + y1_), img.shape[0]),
               max(0, int(bbox_x + x0_)):min(int(bbox_x + x01_), img.shape[1])]
        cut2 = img[max(0, int(bbox_y + y0_)):min(int(bbox_y + y1_), img.shape[0]),
               max(0, int(bbox_x + x10_)):min(int(bbox_x + x1_), img.shape[1])]
        if cut1.shape[0] == 0 or cut1.shape[1] == 0:
            cut1 = np.zeros((1, 1, 3)).astype(np.uint8)
        if cut2.shape[0] == 0 or cut2.shape[1] == 0:
            cut2 = np.zeros((1, 1, 3)).astype(np.uint8)
        return Image.fromarray(cut1),Image.fromarray(cut2)

    
    
    
        
    
    def __repr__(self):
        return self.__class__.__name__




class AA(object):
    def __init__(self):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50, p=0.4),
            albu.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.4),
            albu.GaussianBlur(p=0.3),
            albu.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.9), int(image_size[0])),
                  height=image_size[0],
                  width=image_size[1], p=0.3),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5,border_mode=cv2.BORDER_CONSTANT,p=0.4),
        ]
        self.augs = albu.Compose(augs)   
    
    def __call__(self, image):#, bbox, annotation):
        image = np.array(image)
        image = self.augs(image=image)['image']
        return Image.fromarray(image)


class AA_soft(object):
    def __init__(self):
        augs = [
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50, p=0.4),
            albu.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.4),
            albu.GaussianBlur(p=0.3)]
        self.augs = albu.Compose(augs)   
    
    def __call__(self, image):#, bbox, annotation):
        image = np.array(image)
        image = self.augs(image=image)['image']
        return Image.fromarray(image)





    class MakeTrash(object):
    def __init__(self, data_key = [],target_key = [], final_label=1, trash_size=(10,30), trash_ratio=(0.25,4)):
        '''
        '''
        self.data_key = data_key
        self.target_key = target_key
        self.final_label = final_label
        self.trash_size = trash_size
        self.trash_ratio = trash_ratio
        
    def __call__(self, item_dict):
        item_dict[self.target_key] = self.final_label
        img = item_dict[self.data_key]
        tr = tv.transforms.RandomResizedCrop(img.size,scale=(self.trash_size[0]/img.size[0],self.trash_size[1]/img.size[0]), ratio=self.trash_ratio)
        item_dict[self.data_key] = tr(img)
        return item_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(trash_size={}-{})'.format(self.trash_size[0],self.trash_size[1])



  

class Transform4EachKey(object):
    """
    Apply all torchvision transforms to dict by each key
    """

    def __init__(self, transforms, key_list=['data']):
        self.transforms = transforms
        self.key_list = key_list

    def __call__(self, input_dict):
        for key in self.key_list:
            for t in self.transforms:
                input_dict[key] = t(input_dict[key])
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key_list)
        format_string += '\n)'
        return format_string


