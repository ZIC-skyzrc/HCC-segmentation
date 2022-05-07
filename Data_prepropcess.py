# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:46:16 2022

@author: 1
"""

from __future__ import print_function

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


data_path = '/home3/HWGroup/zhengrc/HCC2021/Train'

image_rows = 256
image_cols = 256
image_depth = 80
patch_depth=16
overlay=8
newshape=np.array([image_depth,image_rows,image_cols], dtype=np.float32)
def create_train_data():
    img_com_list=[]
    mask_com_list=[]
    dirs = sorted(os.listdir(data_path))

    for pat in dirs:
        print(pat)
        imagepath = data_path + "/" + pat + "/" +"co_dyn4.nii"
        maskpath = data_path + "/" + pat + "/" +"l_new.nii"
        image, origin1, spacing1 = load_itk(imagepath)
        m = np.mean(image)
        s = np.std(image)
        image = (image - m) / s

        mask, origin1, spacing1 = load_itk(maskpath)
        image_shape=np.shape(image)
        resize_factor=newshape/[image_shape[0],image_shape[1],image_shape[2]]
        img_resize = nd.interpolation.zoom(image, resize_factor, mode='nearest',order=3)
        msk_resize = nd.interpolation.zoom(mask, resize_factor, mode='nearest',order=0)
        #msk_resize = np.where(msk_resize > 0, 1, 0)
        img_temp_list=[]
        mask_temp_list=[]
        for j in range(0,image_depth,overlay):
            img_temp=img_resize[j:j+overlay,:,:]
            mask_temp=msk_resize[j:j+overlay,:,:]
            img_temp_list.append(img_temp)
            mask_temp_list.append(mask_temp)
        img_temp_list=np.array(img_temp_list)
        mask_temp_list=np.array(mask_temp_list)
        for i in range(0, img_temp_list.shape[0]-1):
            img_com=np.append(img_temp_list[i], img_temp_list[i+1], axis=0)
            mask_com=np.append(mask_temp_list[i], mask_temp_list[i+1], axis=0)
            img_com_list.append(img_com)
            mask_com_list.append(mask_com)
            
    img_com_list=np.array(img_com_list)
    mask_com_list=np.array(mask_com_list)

    imgs = np.expand_dims(img_com_list, axis=4)
    imgs_mask = np.expand_dims(mask_com_list, axis=4) 
    print(np.shape(imgs_mask))
    np.save('./0824/imgs_train_liver.npy', imgs)
    np.save('./0824/mask_train_liver.npy', imgs_mask)


data_path_test ='/home3/HWGroup/zhengrc/HCC2021/Test'
def create_test_data():
    img_com_list=[]

    dirs = sorted(os.listdir(data_path_test))

    for pat in dirs:
        print(pat)
        imagepath = data_path_test + "/" + pat + "/" +"co_dyn4.nii"

        image, origin1, spacing1 = load_itk(imagepath)
        m = np.mean(image)
        s = np.std(image)
        image = (image - m) / s


        image_shape=np.shape(image)
        resize_factor=newshape/[image_shape[0],image_shape[1],image_shape[2]]
        img_resize = nd.interpolation.zoom(image, resize_factor, mode='nearest',order=3)

        #msk_resize = np.where(msk_resize > 0, 1, 0)
        img_temp_list=[]

        for j in range(0,image_depth,overlay):
            img_temp=img_resize[j:j+overlay,:,:]

            img_temp_list.append(img_temp)

        img_temp_list=np.array(img_temp_list)

        for i in range(0, img_temp_list.shape[0]-1):
            img_com=np.append(img_temp_list[i], img_temp_list[i+1], axis=0)

            img_com_list.append(img_com)
  
            
    img_com_list=np.array(img_com_list)
    imgs = np.expand_dims(img_com_list, axis=4)

    np.save('./0824/imgs_test_liver.npy', imgs)
    
 

   
if __name__ == '__main__':
    create_train_data() 
    create_test_data()
    print('finish')