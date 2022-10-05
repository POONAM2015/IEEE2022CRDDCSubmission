# IEEE2022_CRDDC_Submission
The repository contains the source code and trained models for [Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)](https://crddc2022.sekilab.global/overview/) that was held as part of IEEE Big Data Cup 2022.

Maximum F1 score achieved for different leaderboards are as follow:
|Leaderboard |F1 score
|--- |---  
|Combined | 0.697423
|India | 0.493679
|Japan | 0.715857
|Norway | 0.461847
|United States | 0.775131
|Average | 0.628788

<p align="left"><img src="samples_of_predictions.png" width="80%" height="auto"></p>

# Table of contents

- [Steps](#steps)
  - [Download this repo](#download-this-repo)
  - [Copying the data](#copying-the-data)
    - [Training data](#training-data)
    - [Validation data](#validation-data)
    - [Test data](#test-data)
  - [Training the model & Inference](#training-the-model--inference)
    - [Creating and running Docker Image](#creating-and-running-docker-image)
    - [Running the model for training and inference](#running-the-model-for-training-and-inference)
- [References](#references)

# Steps
## Download this repo
Download this repository either using `git clone` or downloading the source code as zip.
After cloning, enter the directory using `cd IEEE2022_CRDDC_Submission`.

## Copying the data
### Training data
- Copy training images of different countries in the `Dataset` in the respective country folder.
- The xml annotations are changed to corresponding yolo annotations.
- Images and labels of a country are kept in `Dataset/[country]/train/images` and `Dataset/[country]/train/labels` respectively.

### Validation data
- Training data is split in the ratio of 80:20 or 85:15 to divide it into training and validation data.

### Test data
- Copy test images of different countries to `Dataset/[country]/test/images`.
- All the test images are also kept in `Dataset/combined_without_drone` for easy inference for `combined` category.

## Training the model & Inference

### Creating and running Docker Image
Make a docker image containing [YOLOv7 source code](https://github.com/WongKinYiu/yolov7) using the command
```bash
docker build -t yolov7:pytorch_2108py3 .
```
Start the docker container using
```bash
nvidia-docker run --rm --name yolov7RDD -p 6006:6006 --ipc=host -it -v "`pwd`/Dataset":/usr/src/mydataset -v "`pwd`/runs":/usr/src/app/yolov7/runs --shm-size=8g yolov7:pytorch_2108py3
```

### Running the model for training and inference
Run [runs/work.ipynb](runs/work.ipynb)

# References
1. https://github.com/WongKinYiu/yolov7 
