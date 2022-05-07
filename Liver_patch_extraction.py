# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:56:43 2021

@author: 1
"""

import os
import SimpleITK as sitk
from scipy import ndimage as nd
import numpy as np



def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def save_itk(image, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)



#data_path='J:/2020project/HCC_final/useforval'
data_path='/home3/HWGroup/zhengrc/HCC2021/Train'
dirs = sorted(os.listdir(data_path))

image_rows = 224
image_cols = 256
image_depth = 64
newshape=np.array([image_depth,image_rows,image_cols], dtype=np.float32)
image_dyn1_list=[]
image_dyn2_list=[]
image_dyn3_list=[]
image_dyn4_list=[]
mask_list=[]
for pat in dirs:
    datapath_dyn1 = data_path + "/" + pat + "/" +"co_dyn1.nii"
    datapath_dyn2 = data_path + "/" + pat + "/" +"co_dyn2.nii"
    datapath_dyn3 = data_path + "/" + pat + "/" +"co_dyn3.nii"
    datapath_dyn4 = data_path + "/" + pat + "/" +"co_dyn4.nii"
    liverpath = data_path + "/" + pat + "/" +"l_new.nii"
    tumorpath = data_path + "/" + pat + "/" +"t_new.nii"

    image_dyn1, origin, spacing = load_itk(datapath_dyn1)
    image_dyn2, origin, spacing = load_itk(datapath_dyn2)
    image_dyn3, origin, spacing = load_itk(datapath_dyn3)
    image_dyn4, origin, spacing = load_itk(datapath_dyn4)
    liver, origin, spacing = load_itk(liverpath)
    tumor, origin, spacing = load_itk(tumorpath)
    liver[liver>0]=1
    tumor[tumor!=1]=0

    
    noposlist=np.where(liver==0)
    image_dyn1[noposlist]=0
    image_dyn2[noposlist]=0
    image_dyn3[noposlist]=0
    image_dyn4[noposlist]=0
    
    m = np.mean(image_dyn1)
    s = np.std(image_dyn1)
    image_dyn1 = (image_dyn1 - m) / s
    image_dyn2 = (image_dyn2 - m) / s
    image_dyn3 = (image_dyn3 - m) / s
    image_dyn4 = (image_dyn4 - m) / s
        
    poslist=np.where(liver>0)
    p1_max=np.max(poslist[0])
    p1_min=np.min(poslist[0])
    p2_max=np.max(poslist[1])
    p2_max=p2_max+10
    p2_min=np.min(poslist[1])
    p2_min=p2_min-10
    p3_max=np.max(poslist[2])
    p3_max=p3_max+10
    p3_min=np.min(poslist[2])
    p3_min=p3_min-10
    image_dyn1_crop=image_dyn1[p1_min:p1_max,p2_min:p2_max,p3_min:p3_max]
    image_dyn2_crop=image_dyn2[p1_min:p1_max,p2_min:p2_max,p3_min:p3_max]
    image_dyn3_crop=image_dyn3[p1_min:p1_max,p2_min:p2_max,p3_min:p3_max]
    image_dyn4_crop=image_dyn4[p1_min:p1_max,p2_min:p2_max,p3_min:p3_max]
    tumor_crop=tumor[p1_min:p1_max,p2_min:p2_max,p3_min:p3_max]
    
    image_shape=np.shape(image_dyn1_crop)
    resize_factor=newshape/[image_shape[0],image_shape[1],image_shape[2]]
    image_dyn1_crop_resize = nd.interpolation.zoom(image_dyn1_crop, resize_factor, mode='nearest',order=3)
    image_dyn2_crop_resize = nd.interpolation.zoom(image_dyn2_crop, resize_factor, mode='nearest',order=3)
    image_dyn3_crop_resize = nd.interpolation.zoom(image_dyn3_crop, resize_factor, mode='nearest',order=3)
    image_dyn4_crop_resize = nd.interpolation.zoom(image_dyn4_crop, resize_factor, mode='nearest',order=3)
    tumor_crop_resize = nd.interpolation.zoom(tumor_crop, resize_factor, mode='nearest',order=0)
    
    image_dyn1_list.append(image_dyn1_crop_resize)
    image_dyn2_list.append(image_dyn2_crop_resize)
    image_dyn3_list.append(image_dyn3_crop_resize)
    image_dyn4_list.append(image_dyn4_crop_resize)
    mask_list.append(tumor_crop_resize)
    print(pat)
    
image_dyn1_list=np.array(image_dyn1_list)
image_dyn1_list = image_dyn1_list.astype('float32')
image_dyn2_list=np.array(image_dyn2_list)
image_dyn2_list = image_dyn2_list.astype('float32')
image_dyn3_list=np.array(image_dyn3_list)
image_dyn3_list = image_dyn3_list.astype('float32')
image_dyn4_list=np.array(image_dyn4_list)
image_dyn4_list = image_dyn4_list.astype('float32')
mask_list=np.array(mask_list)
mask_list = mask_list.astype('float32')

np.save('./0824/image_dyn1_list_train.npy',image_dyn1_list)
np.save('./0824/image_dyn2_list_train.npy',image_dyn2_list)
np.save('./0824/image_dyn3_list_train.npy',image_dyn3_list)
np.save('./0824/image_dyn4_list_train.npy',image_dyn4_list)
np.save('./0824/mask_list_train.npy',mask_list)
