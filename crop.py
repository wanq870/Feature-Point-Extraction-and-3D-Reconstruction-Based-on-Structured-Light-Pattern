import glob
from natsort import natsorted
import os
import cv2
import pandas as pd

def load_crop_params(crop_param_path):
    params = dict()
    with open(crop_param_path, 'r') as f:
        raw_params = f.readline().strip().split(',')
        params['new_row'] = int(raw_params[0])
        params['new_col'] = int(raw_params[1])
        params['old_size'] = int(raw_params[2])
        params['new_size'] = int(raw_params[3])
    return params

def process_images(data_dir, crop_params):
    for dirname in data_dir:
        fnames = natsorted(glob.glob(os.path.join(dirname, '*.png')))
        for fname in fnames:
            img = cv2.imread(fname, 0)
            # crop and resize img
            print(f"Processing image: {fname}")
            img = img[crop_params['new_row']:crop_params['new_row']+crop_params['old_size'], crop_params['new_col']:crop_params['new_col']+crop_params['old_size']]
            img = cv2.resize(img, (crop_params['new_size'], crop_params['new_size']), interpolation=cv2.INTER_AREA)
            
            # save img
            cv2.imwrite(fname, img)
        
def process_labels(label_dir, crop_params):
    for dirname in label_dir:
        fnames = natsorted(glob.glob(os.path.join(dirname, '*.csv')))
        for fname in fnames:
            df = pd.read_csv(fname)
            df = df.iloc[:,1:3]
            
            # crop and resize label
            print(f'Processing label: {fname}')
            df.iloc[:,0] = (df.iloc[:,0] - crop_params['new_row']) * (crop_params['new_size'] / crop_params['old_size'])
            df.iloc[:,1] = (df.iloc[:,1] - crop_params['new_col']) * (crop_params['new_size'] / crop_params['old_size'])
            
            # save label
            df.to_csv(fname)


if __name__ == '__main__':
    crop_param_path = 'crop_params.txt'
    crop_params = load_crop_params(crop_param_path)
    
    dataset = '3dcv_dataset'
    data_dir = [os.path.join(dataset, 'train', 'data'), os.path.join(dataset, 'valid', 'data'), os.path.join(dataset, 'test', 'data')]
    label_dir = [os.path.join(dataset, 'train', 'label'), os.path.join(dataset, 'valid', 'label'), os.path.join(dataset, 'test', 'label')]
    process_images(data_dir, crop_params)
    process_labels(label_dir, crop_params)
        
        