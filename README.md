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
        1.  model_name: *Unet*, *Resnest* or *gt*
        2.  **If the model is *Resnest*, comment line 27 in run.py**
    3. run ./run.sh.
       1. the predicted feature points are saved as **[image_num]**.csv in order in **[model_name]**/csv/ directory
2. Reconstruction:
   1.  make sure "11_coords.csv", "crop_params.txt" and "calibration_result.xml" are in current working directory.
   2.  run ./reconstruct.py **[model_name]** **[img_num]** to see the reconstructed 3D point clouds of **[img_num]**.png.