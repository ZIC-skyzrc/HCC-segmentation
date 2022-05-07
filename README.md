# HCC-segmentation
Automatic Liver Tumor Segmentation on Dynamic Contrast Enhanced MRI Using 4D Information: Deep Learning Model Based on 3D Convolution and Convolutional LSTM

This is the source code for liver tumor segmentation with 3D CNN and Conv-LSTM model.

Data pre-processing for liver segmentation: 
Run python3 Data_prepropcess.py

Liver segmentation with 3D U-net model: 
Run python3 Liver_seg_model.py

Liver patch extraction:
Run python3 Liver_patch_extraction.py

Tumor patch extraction:
Run python3 Tumor_patch_extraction.py

Tumor segmentation with combined CNN and Conv-LSTM model:

2.5D version (2D CNN + Conv-LSTM): Run python3 Tumor_seg_model_2DCT.py
3D version (3D CNN + Conv-LSTM): Run python3 Tumor_seg_model_3DCT.py
