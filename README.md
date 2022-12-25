## NTU-3DCV22-Final-Project
### National Taiwan University CSIE 3D Computer Vision with Deep Learning Final Project in 2022 Fall

#### Models
Three models: U-net, ResNeSt, ResNet based ground truth

#### Testing Data
Many input images **[img_num]**.png are put in 3dcv_dataset/test/data directory.
img_num: 1~N.png


#### How to run
1.  Output from the model:
    1.  uncomment the 6 and 7 line in run.sh.
    2.  modify **[model_name]**.
        1.  model_name: *Unet* or *Resnest*
        2.  **If the model is *Resnest*, comment line 27 in run.py, else uncomment it**
    3. run python3 run.sh.
       1. the predicted feature points are saved as **[image_num]**.csv in order in **[model_name]**/csv/ directory
    4. eg. modify run.sh to below
    python3 run.py 3dcv_dataset Resnest \
    --num_epoch 3500 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    **--ckpt Resnest/model.ckpt** \
    **--do_predict** \
    and run python3 run.sh. You'll see the output of ResNeSt stored in Resnest directory 
2. Reconstruction:
   1.  make sure "11_coords.csv", "crop_params.txt" and "calibration_result.xml" are in current working directory.
   2.  run python3 reconstruct.py **[model_name]** **[img_num]** to see the reconstructed 3D point clouds of **[img_num]**.png
   3.  eg. run python3 reconstruct.py Resnest 1, You'll see the 3D point cloud of 3dcv_dataset/test/1.png