## NTU-3DCV22-Final-Project
### National Taiwan University CSIE 3D Computer Vision with Deep Learning Final Project in 2022 Fall

### Environment
1.  OS: Ubuntu 20.04
2.  GPU: RTX 2080 Ti / A100
3.  Python: 3.9.15

### Packages
1.  torch 1.13.1
2.  torchvision 0.14.1
3.  tqdm 4.64.1
4.  thop 0.1.1.post2209072238
5.  resnest 0.0.6b20221220
6.  pandas 1.5.2
7.  opencv-python 4.6.0.66
8.  numpy 1.23.5
9.  natsort 8.2.0
10. matplotlib 3.6.2
11. glob2 0.7

#### Models
Two models: U-net, ResNeSt

#### Testing Data
image 1.png is put in 3dcv_dataset/test/data directory.



#### How to run
1.  Get the model weights from google drive and unzip the dataset dir
    1.  run download.sh
2.  Output from the model:
    1.  uncomment the 6 and 7 line in run.sh.
    2.  modify **[model_name]** to Unet or Resnest
        1.  **If the model is *Resnest*, comment line 27 in run.py, else uncomment it**
    3. run python3 run.sh.
       1. the predicted feature points are saved as 1.csv in **[model_name]**/csv/
    4. eg. if the model is Resnest, modify run.sh to below:
    python3 run.py 3dcv_dataset Resnest \
    --num_epoch 3500 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    **--ckpt Resnest/model.ckpt** \
    **--do_predict** \
    and run python3 run.sh. You'll see the output of ResNeSt stored in Resnest directory 
3. Reconstruction:
   1.  make sure "11_coords.csv", "crop_params.txt" and "calibration_result.xml" are in current working directory.
   2.  run python3 reconstruct.py **[model_name]** 1 to see the reconstructed 3D point clouds of 1.png
   3.  eg. run python3 reconstruct.py Resnest 1, You'll see the 3D point cloud of 1.png(if you want to see the ground truth point cloud, run python3 reconstruct.py gt 1)