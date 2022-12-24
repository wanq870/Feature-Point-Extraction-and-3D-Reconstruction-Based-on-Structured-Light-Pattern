"""
This reconstructs the 3D point cloud via triangulation

Source files setup:
1. A xml file containing camera and projector parameters
2. 2 csvs of coordinates, 1 for camera's perspective, and another for projector's view.
3. crop_params.txt that contains the transformation parameter for SLResnet
4. Modify the csv names below

Note that the csv of the projector is named as 'xx_coords.csv', xx means the side length of a grid in pixel.

Output:
3D point cloud

How to run:
python3 reconstruct.py
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import argparse
import os

def read_img_params(filepath):
    params = dict()
    with open(filepath, 'r') as f:
        raw_params = f.readline().strip().split(',')
        params['new_row'] = int(raw_params[0])
        params['new_col'] = int(raw_params[1])
        params['old_size'] = int(raw_params[2])
        params['new_size'] = int(raw_params[3])

    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["Unet", "Resnest", 'gt'], help="which directory", type=str)
    parser.add_argument("csv_num", default=1, help="which csv file to reconstruct", type=int)
    args = parser.parse_args()
    
    
    ## Get camera and projector matrices
    fs = cv2.FileStorage(glob.glob('*.xml')[0], cv2.FILE_STORAGE_READ)

    cam_int = fs.getNode('cam_int').mat()
    cam_rvec = cv2.Rodrigues(fs.getNode('cam_rvecs').mat()[-1])[0]
    cam_tvec = fs.getNode('cam_tvecs').mat()[-1].reshape(3,1)
    cam_ext = np.hstack([cam_rvec, cam_tvec])
    cam_proj = np.matmul(cam_int, cam_ext)

    proj_int = fs.getNode('proj_int').mat()
    proj_rvec = cv2.Rodrigues(fs.getNode('proj_rvecs').mat()[-1])[0]
    proj_tvec = fs.getNode('proj_tvecs').mat()[-1].reshape(3,1)
    proj_ext = np.hstack([proj_rvec, proj_tvec])
    proj_proj = np.matmul(proj_int, proj_ext)
    proj_dist = fs.getNode('proj_dist').mat()

    ## Load predicted coordinates
    predict = pd.read_csv(os.path.join(args.model, 'csv', f'{args.csv_num}.csv'))
    predict = predict[['0','1']].to_numpy()
    print(predict)
    # recover the original coordinate system
    crop_params = read_img_params('crop_params.txt')
    predict = predict * (crop_params['old_size'] / crop_params['new_size'])
    predict[:,0] += crop_params['new_row'] 
    predict[:,1] += crop_params['new_col']

    # flip the columns to comply with opencv format
    predict = np.flip(predict,1)
    print(predict)

    # read the grid points in projector's perspective
    coord_gt = pd.read_csv('11_coords.csv') # modify if needed
    coord_gt = coord_gt.iloc[:,1:3].to_numpy().astype(np.float64)
    coord_gt = np.flip(coord_gt, 1)
    coord_gt = cv2.undistortPoints(coord_gt, proj_int, proj_dist, None, proj_int).reshape(-1,2) # Undistort projector's view

    # triangulation
    pt_cloud = cv2.triangulatePoints(cam_proj, proj_proj, predict.T, coord_gt.T)
    pt_cloud = cv2.convertPointsFromHomogeneous(pt_cloud.T).reshape(-1,3)

    x, y, z = pt_cloud[:,0], pt_cloud[:,1], pt_cloud[:,2]

    num_rows = 31
    colors = cm.rainbow(np.linspace(0,1, num_rows))

    ## plot reconstructed 3D point cloud
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(15,180) # set up the perspective of the reconstructed point cloud
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    for i, color in enumerate(colors):
        ax.scatter(x[i*31:i*31+31],y[i*31:i*31+31],z[i*31:i*31+31],color=color)

    plt.show()

