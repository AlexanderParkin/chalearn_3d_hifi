import os, argparse
import sys
import glob
import cv2
from tqdm import tqdm as tqdm
import pandas as pd
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import face_detection

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
    crop = imcrop(img, s_bbox.astype(np.int16))
    return crop


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def write_imgs(arg):
    i, row = arg
    img1 = make_expand_crop_rgb_v2(row, scale_factor=0.3)
    Image.fromarray(img1).save(row['opensource_crop_0.3_path_black'])

def fp(x, d):
    return os.path.join(d,x.path)

def main(args):
    df=pd.read_csv(os.path.join(args.list_dir, args.test_list_out))
    df['full_path'] = df.apply(fp, axis=1, d=args.data_dir)
    df['opensource_crop_0.3_path_black'] = df.apply(fp, axis=1, d=args.crops_dir)
    df.to_csv(os.path.join(args.list_dir, args.test_list_out),index=False)

    for i, row in df.iterrows():
        d = os.path.dirname(row['opensource_crop_0.3_path_black'])
        if not os.path.exists(d):
            os.makedirs(d)

    with Pool(15) as p:
        for res in tqdm(p.imap(write_imgs, df.iterrows()), total=len(df), desc='Processes'):
            if res is not None:
                pass
    print('test imgs done')

    df=pd.read_csv(os.path.join(args.list_dir, args.val_list_out))
    df['full_path'] = df.apply(fp, axis=1, d=args.data_dir)
    df['opensource_crop_0.3_path_black'] = df.apply(fp, axis=1, d=args.crops_dir)
    df.to_csv(os.path.join(args.list_dir, args.val_list_out),index=False)

    for i, row in df.iterrows():
        d = os.path.dirname(row['opensource_crop_0.3_path_black'])
        if not os.path.exists(d):
            os.makedirs(d)

    with Pool(15) as p:
        for res in tqdm(p.imap(write_imgs, df.iterrows()), total=len(df), desc='Processes'):
            if res is not None:
                pass
    print('val imgs done')

    df=pd.read_csv(os.path.join(args.list_dir, args.train_list_out))
    df['full_path'] = df.apply(fp, axis=1, d=args.data_dir)
    df['opensource_crop_0.3_path_black'] = df.apply(fp, axis=1, d=args.crops_dir)
    df.to_csv(os.path.join(args.list_dir, args.train_list_out),index=False)

    for i, row in df.iterrows():
        d = os.path.dirname(row['opensource_crop_0.3_path_black'])
        if not os.path.exists(d):
            os.makedirs(d)

    with Pool(15) as p:
        for res in tqdm(p.imap(write_imgs, df.iterrows()), total=len(df), desc='Processes'):
            if res is not None:
                pass
    print('train imgs done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rgb+ir liveness')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path to the directory, where all train/test/val raw images are located.')
    parser.add_argument('--list_dir',
                        type=str, default = '.',
                        help='Path to the directory, where all lists are located.')
    parser.add_argument('--test_list_out',
                        type=str,
                        help='Name to save the final test list.')
    parser.add_argument('--train_list_out',
                        type=str,
                        help='Name to save the final train list.')
    parser.add_argument('--val_list_out',
                        type=str,
                        help='Name to save the final val list.')
    parser.add_argument('--crops_dir',
                        type=str,
                        help='Path to the directory, where all processed images will be located.')

    args = parser.parse_args()
    main(args)


