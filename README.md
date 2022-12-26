
## NTU-3DCV22-Final-Project
### Feature Point Extraction and 3D Reconstruction Based on Structured-Light Pattern

### Hardwares
1.  CPU: R9 7950X / EPYC 7742
2.  GPU: RTX 2080 Ti / A100

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
 
### Enviroment
```shell
pip install -r requirements.txt
```

### Download pretrained models
```shell
# Download our pretrained U-net, ResNest models
bash download.sh
```

### Train 
```shell
# Train with default Resnest
bash train.sh
```

### Test (Predict)
Please make sure that 1.png is in 3dcv_dataset/test/data directory.
```shell
# Predict with our pretrained model, with default Resnest
bash test.sh
```

### Change Model
#### 1. Change Resnest to Unet in *.sh files.
#### 2. Uncomment line 24 and 28 in run.py.

### Reconstruction:
Please make sure that "11_coords.csv", "crop_params.txt" and "calibration_result.xml" are in current working directory.
```shell
python3 reconstruct.py Resnest 1
```
You can change "Resnest" to gt or Unet to reconstruct 3D point cloud with other model.
