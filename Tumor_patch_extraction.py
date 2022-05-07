# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:28 2020

@author: 1
"""

import numpy as np

### Training and Validation set
image_dyn1_train=np.load('./0824/image_dyn1_list_train.npy')
image_dyn2_train=np.load('./0824/image_dyn2_list_train.npy')
image_dyn3_train=np.load('./0824/image_dyn3_list_train.npy')
image_dyn4_train=np.load('./0824/image_dyn4_list_train.npy')

mask_train=np.load('./0824/mask_list_train.npy')

patch_dyn1_list=[]
patch_dyn2_list=[]
patch_dyn3_list=[]
patch_dyn4_list=[]
mask_list=[]

for i in range(np.size(mask_train,0)):
    tumor_slice=np.where(mask_train[i]==1)
    tumor_slice_max=np.max(tumor_slice[0])
    tumor_slice_min=np.min(tumor_slice[0])
    for j in range(np.max([3,tumor_slice_min-3]),np.min([tumor_slice_max+4,np.size(mask_train[i],0)-3])):
        pos_patch_dyn1=image_dyn1_train[i,j-3:j+4,:,:]
        pos_patch_dyn2=image_dyn2_train[i,j-3:j+4,:,:]
        pos_patch_dyn3=image_dyn3_train[i,j-3:j+4,:,:]
        pos_patch_dyn4=image_dyn4_train[i,j-3:j+4,:,:]
        pos_mask=mask_train[i,j,:,:]
        patch_dyn1_list.append(pos_patch_dyn1)
        patch_dyn2_list.append(pos_patch_dyn2)
        patch_dyn3_list.append(pos_patch_dyn3)
        patch_dyn4_list.append(pos_patch_dyn4)
        mask_list.append(pos_mask)

    for j in range(3,tumor_slice_min-3,3):
        neg_patch_dyn1=image_dyn1_train[i,j-3:j+4,:,:]
        neg_patch_dyn2=image_dyn2_train[i,j-3:j+4,:,:]
        neg_patch_dyn3=image_dyn3_train[i,j-3:j+4,:,:]
        neg_patch_dyn4=image_dyn4_train[i,j-3:j+4,:,:]
        neg_mask=mask_train[i,j,:,:]
        patch_dyn1_list.append(neg_patch_dyn1)
        patch_dyn2_list.append(neg_patch_dyn2)
        patch_dyn3_list.append(neg_patch_dyn3)
        patch_dyn4_list.append(neg_patch_dyn4)
        mask_list.append(neg_mask)

    for j in range(tumor_slice_max+4,np.size(mask_train[i],0)-3,3):
        neg_patch_dyn1=image_dyn1_train[i,j-3:j+4,:,:]
        neg_patch_dyn2=image_dyn2_train[i,j-3:j+4,:,:]
        neg_patch_dyn3=image_dyn3_train[i,j-3:j+4,:,:]
        neg_patch_dyn4=image_dyn4_train[i,j-3:j+4,:,:]
        neg_mask=mask_train[i,j,:,:]
        patch_dyn1_list.append(neg_patch_dyn1)
        patch_dyn2_list.append(neg_patch_dyn2)
        patch_dyn3_list.append(neg_patch_dyn3)
        patch_dyn4_list.append(neg_patch_dyn4)
        mask_list.append(neg_mask)
    print(i)

patch_dyn1_list=np.array(patch_dyn1_list) 
patch_dyn2_list=np.array(patch_dyn2_list) 
patch_dyn3_list=np.array(patch_dyn3_list) 
patch_dyn4_list=np.array(patch_dyn4_list) 
mask_list=np.array(mask_list)

patch_dyn1_list = patch_dyn1_list.astype('float32')
patch_dyn2_list = patch_dyn2_list.astype('float32')
patch_dyn3_list = patch_dyn3_list.astype('float32')
patch_dyn4_list = patch_dyn4_list.astype('float32')
mask_list=mask_list.astype('float32')

print(patch_dyn1_list.shape)
print(patch_dyn2_list.shape)
print(patch_dyn3_list.shape)
print(patch_dyn4_list.shape)
print(mask_list.shape)

np.save('./0824/patch_dyn1_list_train.npy',patch_dyn1_list)   
np.save('./0824/patch_dyn2_list_train.npy',patch_dyn2_list)     
np.save('./0824/patch_dyn3_list_train.npy',patch_dyn3_list)     
np.save('./0824/patch_dyn4_list_train.npy',patch_dyn4_list)
np.save('./0824/patch_mask_list_train.npy',mask_list)   


### Test set
image_dyn1_train=np.load('./0824/image_dyn1_list_test.npy')
image_dyn2_train=np.load('./0824/image_dyn2_list_test.npy')
image_dyn3_train=np.load('./0824/image_dyn3_list_test.npy')
image_dyn4_train=np.load('./0824/image_dyn4_list_test.npy')

mask_train=np.load('./0824/mask_list_test.npy')

patch_dyn1_list=[]
patch_dyn2_list=[]
patch_dyn3_list=[]
patch_dyn4_list=[]
mask_list=[]

min_slice=3
max_slice=np.size(mask_train,1)-3
for i in range(np.size(mask_train,0)):
    for j in range(min_slice,max_slice):
        pos_patch_dyn1=image_dyn1_train[i,j-3:j+4,:,:]
        pos_patch_dyn2=image_dyn2_train[i,j-3:j+4,:,:]
        pos_patch_dyn3=image_dyn3_train[i,j-3:j+4,:,:]
        pos_patch_dyn4=image_dyn4_train[i,j-3:j+4,:,:]
        pos_mask=mask_train[i,j,:,:]
        patch_dyn1_list.append(pos_patch_dyn1)
        patch_dyn2_list.append(pos_patch_dyn2)
        patch_dyn3_list.append(pos_patch_dyn3)
        patch_dyn4_list.append(pos_patch_dyn4)
        mask_list.append(pos_mask)

    print(i)

patch_dyn1_list=np.array(patch_dyn1_list) 
patch_dyn2_list=np.array(patch_dyn2_list) 
patch_dyn3_list=np.array(patch_dyn3_list) 
patch_dyn4_list=np.array(patch_dyn4_list) 
mask_list=np.array(mask_list)   
print(patch_dyn1_list.shape)
print(patch_dyn2_list.shape)
print(patch_dyn3_list.shape)
print(patch_dyn4_list.shape)
print(mask_list.shape)

patch_dyn1_list = patch_dyn1_list.astype('float32')
patch_dyn2_list = patch_dyn2_list.astype('float32')
patch_dyn3_list = patch_dyn3_list.astype('float32')
patch_dyn4_list = patch_dyn4_list.astype('float32')
mask_list=mask_list.astype('float32')

np.save('./0824/patch_dyn1_list_test.npy',patch_dyn1_list)   
np.save('./0824/patch_dyn2_list_test.npy',patch_dyn2_list)     
np.save('./0824/patch_dyn3_list_test.npy',patch_dyn3_list)     
np.save('./0824/patch_dyn4_list_test.npy',patch_dyn4_list)
np.save('./0824/patch_mask_list_test.npy',mask_list)           
  