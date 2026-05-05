# sEMG-based-3D-hand-gesture-prediction-using-Transformer-Encoder-Classifier
3D hand gesture prediction using custom Transformer Encoder Classifier  using sEMG signals acquired from Cyton Biosensing Board and 3D keypoints via LEAP motion IR controller.


Citation: Mahboob, T., Chung, M. Y., & Choi, K. W. (2023). EMG-based 3D hand gesture prediction using transformer–encoder classification. ICT Express, 9(6), 1047-1052.
- Open Access URL for the research journal paper: https://www.sciencedirect.com/science/article/pii/S240595952300053X

IEEE Dataport: https://ieee-dataport.org/documents/semg-8-channel-data-using-openbci-cyton-biosensing-board-and-3d-keypoints-using-leap

![Process_Flow]https://github.com/tmahboob/sEMG_3D_Transofmer_Encoder_Classifier/blob/main/Process Flow.jpg

# What is EMG-prediction?
sEMG_3D_TEC is a set of DL models used for predicting 3D hand postures trained on sEMG channels as inputs and prediction 3D keypoints showing 3D hand gestures. We present a 3D hand gesture prediction application, leveraging the sEMG signal and the optical hand tracking information. A transformer–encoder classifier (TEC) module is introduced in an IPC-system to predict the 3D-hand gestures using eight monopolar channels from sEMG as input. An experimental testbed is setup to acquire, train, and predict the 3D hand gestures within a feasible range of performance. The performance has been evaluated in terms of percentage of correctly classified keypoints (PCK). PCK is measured by first estimating the euclidean distance between the actual and the predicted keypoints. The percentage of keypoints within a threshold distance value are then calculated.

Real data is acquired form two sources: sEMG channel data using 1) Cyton + Daisy Biosensing Boards (16-Channels) 2) Leap motion controller 

https://shop.openbci.com/products/cyton-daisy-biosensing-boards-16-channel?srsltid=AfmBOorZEXvGr_Rq4PhI-1YwozCPYsny9c8cQMCw_-qEn2YKxJGGTtg0 
https://www.ultraleap.com/products/


# Installation
The script can be run natively on IDE such as PyCharm IDE or jupyter notebook with requirements installed
Download the Main Script(sEMG_prediction), dataset(OpenBCI_leap.csv), and the requirements.txt files.

*pip install -r requirements.txt

## Dataset demographics:
* Column 1: Index
* Column 2-9: Channel 1 to Channel 8 sEMG measurements from OpenBCI biosensing board
* Column 10-27: 3D keypoints for the finger tips and the center of the palm collected via Leap motion IR controller

## Input of 3D keypoint prediction is the Biosensing data (ch1-ch8) and prediction is the 3D hand gestures

## Scripts:
* sEMg-prediction.py: 3D hand gesture prediction from sEMG input
* RESNET.py: resnet-50 implementation
* UNET.py: UNET implementation
* Data-filtered_50(CNN).py: CNN model test on sEMG





