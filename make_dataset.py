import os
import sys
import glob
import cv2
from tqdm import tqdm as tqdm
import pandas as pd
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

list_dir = '/media3/new_homes/e.glazistova/media/cha_data/lists/'
train_crops_dir = '/media3/new_homes/e.glazistova/media/rep/crops_train/'
val_crops_dir = '/media3/new_homes/e.glazistova/media/rep/crops_val/'
test_crops_dir = '/media3/new_homes/e.glazistova/media/rep/crops_test/'
test_file = 'combined_test_v1_164.txt'
train_file = 'train_annotated_164.txt'
val_file = 'val_annotated_v2_164.txt'
test_file_out = 'combined_test_v1_detections_bb_164.txt'
train_file_out = 'train_annotated_bb_164.txt'
val_file_out = 'val_annotated_v2_bb_164.txt'

df = pd.read_csv(list_dir+test_file)

i = 0
n = 0
nums = [30000, 30000, 30000, 13998, 16002, 30000, 6998, 21267]
while i<len(df):
    df_part = df[i:min(i+nums[n],len(df))]
    df_part.to_csv(list_dir+test_file+'p'+str(n)+'_.txt',index=False)
    i+=nums[n]
    n+=1

import face_detection

print(face_detection.available_detectors)
detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


def find_boxes(arg):
    # i,arg = arg
    if True:
        pil_img = Image.open(arg['full_path']).convert('RGB')
        img = np.array(pil_img)
        res = detector.detect(img)
        if len(res) == 0:
            return arg
        area = 0
        biggest_res = None
        for r in res:
            if (r[2] - r[0]) * (r[3] - r[1]) > area:
                area = (r[2] - r[0]) * (r[3] - r[1])
                biggest_res = r
        arg['opensource_detection_score'] = biggest_res[4]
        arg['opensource_bbox_x'] = biggest_res[0]
        arg['opensource_bbox_y'] = biggest_res[1]
        arg['opensource_bbox_w'] = biggest_res[2] - biggest_res[0]
        arg['opensource_bbox_h'] = biggest_res[3] - biggest_res[1]
        return arg


def find_boxes_big(arg):
    # i,arg = arg
    if True:
        pil_img = Image.open(arg['full_path']).convert('RGB')
        img = np.array(pil_img)
        if max(img.shape[0], img.shape[1]) > 640:
            k = 640 / max(img.shape[0], img.shape[1])
        else:
            k=1
        img = cv2.resize(img, (0, 0), fx=k, fy=k)
        res = detector.detect(img)
        if len(res) == 0:
            return arg
        area = 0
        biggest_res = None
        for r in res:
            if (r[2] - r[0]) * (r[3] - r[1]) > area:
                area = (r[2] - r[0]) * (r[3] - r[1])
                biggest_res = r
        arg['opensource_detection_score'] = biggest_res[4]
        arg['opensource_bbox_x'] = biggest_res[0] / k
        arg['opensource_bbox_y'] = biggest_res[1] / k
        arg['opensource_bbox_w'] = (biggest_res[2] - biggest_res[0]) / k
        arg['opensource_bbox_h'] = (biggest_res[3] - biggest_res[1]) / k
        return arg


for n in range(7):
    df = pd.read_csv(list_dir + test_file + 'p' + str(n) + '_.txt')

    df['opensource_detection_score'] = None
    df['opensource_bbox_x'] = None
    df['opensource_bbox_y'] = None
    df['opensource_bbox_w'] = None
    df['opensource_bbox_h'] = None

    data = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i, len(df))
        data.append(find_boxes(row))
    df1 = pd.DataFrame(data)
    df1.to_csv(list_dir + test_file + 'p' + str(n) + '_bb.txt', index=False)

n = 7
df = pd.read_csv(list_dir + test_file + 'p' + str(n) + '_.txt')

df['opensource_detection_score'] = None
df['opensource_bbox_x'] = None
df['opensource_bbox_y'] = None
df['opensource_bbox_w'] = None
df['opensource_bbox_h'] = None

data = []
for i, row in df.iterrows():
    if i % 1000 == 0:
        print(i, len(df))
    data.append(find_boxes_big(row))
df1 = pd.DataFrame(data)
df1.to_csv(list_dir + test_file + 'p' + str(n) + '_bb.txt', index=False)

dfs = []
for n in range(8):
    dfs.append(pd.read_csv(list_dir + test_file + 'p' + str(n) + '_bb.txt'))

dfs = pd.concat(dfs, axis=0, join='outer', join_axes=None, ignore_index=True,  # ,train_a
                keys=None, levels=None, names=None, verify_integrity=False,
                copy=True)
dfs.to_csv(list_dir + test_file_out, index=False)
print('test detections done!')


df = pd.read_csv(list_dir + val_file)

df['opensource_detection_score'] = None
df['opensource_bbox_x'] = None
df['opensource_bbox_y'] = None
df['opensource_bbox_w'] = None
df['opensource_bbox_h'] = None

data = []
for i, row in df.iterrows():
    if i % 1000 == 0:
        print(i, len(df))
    data.append(find_boxes(row))
df1 = pd.DataFrame(data)
df1.to_csv(list_dir + val_file_out, index=False)
print('val detections done!')

df = pd.read_csv(list_dir + train_file)

df['opensource_detection_score'] = None
df['opensource_bbox_x'] = None
df['opensource_bbox_y'] = None
df['opensource_bbox_w'] = None
df['opensource_bbox_h'] = None

data = []
for i, row in df.iterrows():
    if i % 1000 == 0:
        print(i, len(df))
    data.append(find_boxes(row))
df1 = pd.DataFrame(data)
df1.to_csv(list_dir + train_file_out, index=False)
print('train detections done!')


def make_expand_crop_rgb_v2(row, scale_factor=0.3):
    img = np.array(Image.open(row.full_path).convert('RGB'))
    if pd.isna(row.opensource_bbox_x):
        return img
    bbox = (row.opensource_bbox_x, row.opensource_bbox_y,
            row.opensource_bbox_x + row.opensource_bbox_w, row.opensource_bbox_y + row.opensource_bbox_h)

    width = row.opensource_bbox_w
    height = row.opensource_bbox_h
    wh = (width, height)

    max_side = max(wh)
    dwh = ((max_side - width) // 2, (max_side - height) // 2)
    rwh = ((max_side - width) % 2, (max_side - height) % 2)
    d_bbox = np.array([-dwh[0] - rwh[0], -dwh[1] - rwh[1], dwh[0], dwh[1]])
    s_bbox = bbox + d_bbox
    d_bbox = np.array(
        [-max_side * scale_factor, -max_side * scale_factor, max_side * scale_factor, max_side * scale_factor])
    s_bbox = s_bbox + d_bbox
    # expand crop
    crop = imcrop(img, s_bbox.astype(np.int16))
    # resized_crop = cv2.resize(crop,(crop_size,crop_size))
    return crop


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    # print('0',y1,y2, x1,x2,img.shape)
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    # print(y1,y2, x1,x2,img.shape)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT)
    # print('mins',-min(0, y1),-min(0, x1),img is None)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    # print('pad',y1,y2, x1,x2,img.shape)
    return img, x1, x2, y1, y2


df=pd.read_csv(list_dir+test_file_out)
def path(x):
    return test_crops_dir+x.full_path.replace('/','~')+'_black.png'

df['opensource_crop_0.3_path_black'] = df.apply(path,axis=1)
df.to_csv(list_dir+test_file_out,index=False)


def write_imgs(arg):
    i, row = arg
    img1 = make_expand_crop_rgb_v2(row, scale_factor=0.3)
    Image.fromarray(img1).save(row['opensource_crop_0.3_path_black'])


if not os.path.exists(test_crops_dir):
    os.makedirs(test_crops_dir)

with Pool(15) as p:
    for res in tqdm(p.imap(write_imgs, df.iterrows()), total=len(df), desc='Processes'):
        if res is not None:
            pass
print('test imgs done')


df=pd.read_csv(list_dir+val_file_out)

def path(x):
    return val_crops_dir+x.full_path.replace('/','~')+'_black.png'

df['opensource_crop_0.3_path_black'] = df.apply(path,axis=1)
df.to_csv(list_dir+val_file_out, index=False)


if not os.path.exists(val_crops_dir):
    os.makedirs(val_crops_dir)

with Pool(15) as p:
    for res in tqdm(p.imap(write_imgs, df.iterrows()), total=len(df), desc='Processes'):
        if res is not None:
            pass
print('val imgs done')

df=pd.read_csv(list_dir+train_file_out)
def path(x):
    return train_crops_dir+x.full_path.replace('/','~')+'_black.png'

df['opensource_crop_0.3_path_black'] = df.apply(path,axis=1)
df.to_csv(list_dir+train_file_out, index=False)

if not os.path.exists(train_crops_dir):
    os.makedirs(train_crops_dir)

with Pool(15) as p:
    for res in tqdm(p.imap(write_imgs, df.iterrows()), total=len(df), desc='Processes'):
        if res is not None:
            pass
print('train imgs done')
