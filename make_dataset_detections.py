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

def find_boxes(arg, detector):
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


def find_boxes_big(arg, detector):
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

def fp(x, d):
    return os.path.join(d,x.path)

def main(args):
    df = pd.read_csv(os.path.join(args.list_dir, args.test_list), header=None, names=['path'])
    df1 = pd.read_csv(os.path.join(args.list_dir, args.val_list), header=None, names=['path', 'label'], sep=' ')
    df1= df1[['path']]

    df1['full_path'] = df1.apply(fp, axis=1, d=args.data_dir)
    df['full_path'] = df.apply(fp, axis=1, d=args.data_dir)

    df = pd.concat([df1, df], axis=0, join='outer', ignore_index=True,  # ,train_a
                     keys=None, levels=None, names=None, verify_integrity=False,
                     copy=True)

    i = 0
    n = 0
    nums = [30000,30000,30000,13998,16002,30000,6998,21267]
    while i<len(df):
        df_part = df[i:min(i+nums[n],len(df))]
        df_part.to_csv(os.path.join(args.list_dir,args.test_list+'p'+str(n)+'_.txt'),index=False)
        i+=nums[n]
        n+=1

    print(face_detection.available_detectors)
    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    for n in range(7):
        df = pd.read_csv(os.path.join(args.list_dir, args.test_list + 'p' + str(n) + '_.txt'))

        df['opensource_detection_score'] = None
        df['opensource_bbox_x'] = None
        df['opensource_bbox_y'] = None
        df['opensource_bbox_w'] = None
        df['opensource_bbox_h'] = None

        data = []
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(i, len(df))
            data.append(find_boxes(row, detector))
        df1 = pd.DataFrame(data)
        df1.to_csv(os.path.join(args.list_dir, args.test_list + 'p' + str(n) + '_bb.txt'), index=False)

    n = 7
    df = pd.read_csv(os.path.join(args.list_dir, args.test_list + 'p' + str(n) + '_.txt'))

    df['opensource_detection_score'] = None
    df['opensource_bbox_x'] = None
    df['opensource_bbox_y'] = None
    df['opensource_bbox_w'] = None
    df['opensource_bbox_h'] = None

    data = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i, len(df))
        data.append(find_boxes_big(row, detector))
    df1 = pd.DataFrame(data)
    df1.to_csv(os.path.join(args.list_dir, args.test_list + 'p' + str(n) + '_bb.txt'), index=False)

    dfs = []
    for n in range(8):
        dfs.append(pd.read_csv(os.path.join(args.list_dir, args.test_list + 'p' + str(n) + '_bb.txt')))

    dfs = pd.concat(dfs, axis=0, join='outer', join_axes=None, ignore_index=True,  # ,train_a
                    keys=None, levels=None, names=None, verify_integrity=False,
                    copy=True)
    dfs.to_csv(os.path.join(args.list_dir, args.test_list_out), index=False)
    print('test detections done!')

    df = pd.read_csv(os.path.join(args.list_dir, args.val_list), header=None, names=['path', 'label'], sep=' ')
    df['full_path'] = df.apply(fp, axis=1, d=args.data_dir)

    df['opensource_detection_score'] = None
    df['opensource_bbox_x'] = None
    df['opensource_bbox_y'] = None
    df['opensource_bbox_w'] = None
    df['opensource_bbox_h'] = None

    data = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i, len(df))
        data.append(find_boxes(row, detector))
    df1 = pd.DataFrame(data)
    df1.to_csv(os.path.join(args.list_dir, args.val_list_out), index=False)
    print('val detections done!')

    df = pd.read_csv(os.path.join(args.list_dir, args.train_list), header=None, names=['path', 'label'], sep=' ')
    df['full_path'] = df.apply(fp, axis=1, d=args.data_dir)

    df['opensource_detection_score'] = None
    df['opensource_bbox_x'] = None
    df['opensource_bbox_y'] = None
    df['opensource_bbox_w'] = None
    df['opensource_bbox_h'] = None

    data = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i, len(df))
        data.append(find_boxes(row, detector))
    df1 = pd.DataFrame(data)
    df1.to_csv(os.path.join(args.list_dir, args.train_list_out), index=False)
    print('train detections done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rgb+ir liveness')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path to the directory, where all train/test/val raw images are located.')
    parser.add_argument('--list_dir',
                        type=str, default = '.',
                        help='Path to the directory, where all lists are located.')
    parser.add_argument('--test_list',
                        type=str,
                        help='Name of a file, where test relative paths are written.')
    parser.add_argument('--train_list',
                        type=str,
                        help='Name of a file, where train relative paths and labels are written.')
    parser.add_argument('--val_list',
                        type=str,
                        help='Name of a file, where val relative paths and labels are written.')
    parser.add_argument('--test_list_out',
                        type=str,
                        help='Name to save the final test list.')
    parser.add_argument('--train_list_out',
                        type=str,
                        help='Name to save the final train list.')
    parser.add_argument('--val_list_out',
                        type=str,
                        help='Name to save the final val list.')

    args = parser.parse_args()
    main(args)


