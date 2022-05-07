# HCC-segmentation
Automatic Liver Tumor Segmentation on Dynamic Contrast Enhanced MRI Using 4D Information: Deep Learning Model Based on 3D Convolution and Convolutional LSTM

This is the source code for liver tumor segmentation with 3D CNN and Conv-LSTM model.


1. Data pre-processing for liver segmentation:

  Run python3 Data_prepropcess.py


2. Data pre-processing for liver segmentation:

  Run python3 Data_prepropcess.py


3. Liver patch extraction:
  
  Run python3 Liver_patch_extraction.py


4. Tumor patch extraction:
  
  Run python3 Tumor_patch_extraction.py


5. Tumor segmentation with combined CNN and Conv-LSTM model:
 
  version (2D CNN + Conv-LSTM): Run python3 Tumor_seg_model_2DCT.py
  
  version (3D CNN + Conv-LSTM): Run python3 Tumor_seg_model_3DCT.py
