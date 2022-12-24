import glob
import os
from natsort import natsorted
import random
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# def load_cam_params(filepath):
#     fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
#     cam_int = fs.getNode('cam_int').mat()
#     cam_dist = fs.getNode('cam_dist').mat()

#     return (cam_int, cam_dist)

# cam_param_path = glob.glob(os.path.join('cache/*.xml'))
# cam_param = load_cam_params(cam_param_path[0])

# label_paths = natsorted(glob.glob(os.path.join('label/*.csv')))
# for label_path in label_paths:
#     if 'update' in str(label_path):
#         new_path = str(label_path).replace('_update', '')
#         os.rename(label_path, new_path)
#         label_path = new_path
# data_paths = natsorted(glob.glob(os.path.join('data/*.pgm')))

# idx_list = []
# for label_path in label_paths:
#     idx = label_path[6:-4]
#     idx_list.append(idx)
# for data_path in data_paths:
#     if (str(data_path)[5:-4]) in idx_list:
#         img = cv2.imread(data_path)
#         img = cv2.undistort(img, cam_param[0], cam_param[1])
#         cv2.imwrite(os.path.join('undistorted_data',f'{str(data_path)[5:-4]}.png'), img)
#     else:
#         os.remove(data_path)
#         data_paths.remove(data_path)

# for idx, (label, data) in enumerate(zip(label_paths, data_paths)):
#     os.rename(label, f'label/{idx+1}.csv')
#     os.rename(data, f'data/{idx+1}.pgm')
#     label_paths[idx] = f'label/{idx+1}.csv'
#     data_paths[idx] = f'data/{idx+1}.pgm'

# label_paths = natsorted(glob.glob(os.path.join('label/*.csv')))
# data_paths = natsorted(glob.glob(os.path.join('undistorted_data/*.png')))

# img_list = []
# label_list = []
# for idx, (label_path, data_path) in enumerate(zip(label_paths, data_paths)):
#     df = pd.read_csv(label_path)
#     df = df.iloc[:,1:3]
#     label_list.append(df)
#     img = cv2.imread(data_path)
#     img_list.append(img)
# size = img_list[0].shape

# for i in range(7):
#     chosen_idx = random.sample(range(1, 72), 20)

#     random_x = int(size[0] / 2 + 100 * random.gauss(mu=0, sigma=1))
#     random_y = int(size[1] / 2 + 100 * random.gauss(mu=0, sigma=1))
#     radius = random.randrange(13, 23, 2)
#     move = random.choice([True, False])
#     for idx in chosen_idx:
#         if  move:
#             img_ = img_list[idx].copy() 
#             img_ = cv2.circle(img_, (random_y, random_x), radius, (0, 0, 0), -1)
#         else:
#             img_ = img_list[idx].copy() 
#             img_ = cv2.circle(img_, (random_y, random_x), radius, (255, 255, 255), -1)
#         img_list.append(img_)
#         label_list.append(label_list[idx])

# train_id = 1
# valid_id = 1
# for idx, (label, img) in enumerate(zip(label_list, img_list)):
#     if (idx+1) % 10 <= 2:
#         cv2.imwrite(os.path.join(f'valid/data/{valid_id}.png'), img)
#         label.to_csv(os.path.join(f'valid/label/{valid_id}.csv'))
#         valid_id += 1
#     else:
#         cv2.imwrite(os.path.join(f'train/data/{train_id}.png'), img)
#         label.to_csv(os.path.join(f'train/label/{train_id}.csv'))
#         train_id += 1
#     print(f'Data {idx} finished.')

train_paths = natsorted(glob.glob(os.path.join('3dcv_dataset/train/label/*.csv')))
valid_paths = natsorted(glob.glob(os.path.join('3dcv_dataset/valid/label/*.csv')))
test_paths = natsorted(glob.glob(os.path.join('3dcv_dataset/test/label/*.csv')))
paths = [train_paths, valid_paths, test_paths]
for path in paths:
    for label_path in path:    
        print(label_path)
        df = pd.read_csv(label_path)
        df = df.iloc[:,1:3]
        df = df-720
        df.to_csv(os.path.join(label_path))