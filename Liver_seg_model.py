# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:36:09 2022

@author: 1
"""

import os
import keras.models as models
import numpy as np

#np.random.seed(256)
import tensorflow as tf
#tf.set_random_seed(256)

from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
import math

#from data3D import load_train_data,load_test_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

K.set_image_data_format('channels_last')

project_name = '3D-Unet_Liver_seg'
img_rows = 256
img_cols = 256
img_depth = 16
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_depth, img_rows, img_cols, 1))
    conv1 = Conv3D(16, (3, 3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(16, (3, 3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    

    conv2 = Conv3D(32, (3, 3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(32, (3, 3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(64, (3, 3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, (3, 3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(128, (3, 3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(256, (3, 3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(256, (3, 3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
    up6 = BatchNormalization()(up6)
    conv6 = Conv3D(128, (3, 3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(128, (3, 3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    up7 = BatchNormalization()(up7)
    conv7 = Conv3D(64, (3, 3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(64, (3, 3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    up8 = BatchNormalization()(up8)
    conv8 = Conv3D(32, (3, 3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv3D(32, (3, 3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    up9 = BatchNormalization()(up9)
    conv9 = Conv3D(16, (3, 3, 3), padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv3D(16, (3, 3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()


    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00000199), loss=dice_coef_loss, metrics=['accuracy'])

    return model


def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 100.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate

def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train=np.load('./0824/imgs_train_liver.npy')
    print('imgs_train:',imgs_train.shape)

    imgs_mask_train=np.load('./0824/mask_train_liver.npy')
    print('imgs_mask_train:',imgs_mask_train.shape)
    
    imgs_val=np.load('./0824/imgs_val_liver.npy')    
    print('imgs_val:',imgs_val.shape)
    
    imgs_mask_val=np.load('./0824/mask_val_liver.npy')
    print('imgs_mask_val:',imgs_mask_val.shape)


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model=get_unet()
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    learning_rate = 0.001
    model.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00000199), loss=dice_coef_loss, metrics=['accuracy'])
    
    
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + '.h5'), monitor='val_loss', save_best_only=True)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + '.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    

    model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=200, verbose=2, shuffle=True, validation_data=(imgs_val,imgs_mask_val), callbacks=[model_checkpoint, csv_logger,lrate])


    print('-'*30)
    print('Training finished')
    print('-'*30)

def predict():

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

        
    imgs_test=np.load('./0824/imgs_test_liver.npy')    
    print('imgs_test:',imgs_test.shape)
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)

    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    print(imgs_mask_test.shape)

    npy_mask_dir = 'test_mask_npy'
    if not os.path.exists(npy_mask_dir):
        os.mkdir(npy_mask_dir)

    np.save(os.path.join(npy_mask_dir, project_name + '_liver_mask.npy'), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    
if __name__ == '__main__':
    train()
    predict()
