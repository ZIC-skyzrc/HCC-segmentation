# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 00:59:51 2021

@author: 1
"""

import os
import keras.models as models
import numpy as np

np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(0)

from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
import math

#from data3D import load_train_data,load_test_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

K.set_image_data_format('channels_last')

project_name = '2DCT_0824'
img_rows = 224
img_cols = 256
img_depth = 7
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def backend_exdims(x):
    return K.expand_dims(x, -1)

def backend_transpose(x):
    return tf.transpose(x, perm=[0,4,1,2,3])

def single_unet(dyn_input):
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(dyn_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    up1 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    up1 = BatchNormalization()(up1)
    
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    up2 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
    up2 = BatchNormalization()(up2)
    
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='valid')(up2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='valid')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='valid')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Lambda(backend_exdims)(conv5)
    
    return conv5
    

def conv_clstm():
    inputs1 = Input((img_rows, img_cols, img_depth),name="dyn1")
    inputs2 = Input((img_rows, img_cols, img_depth),name="dyn2")
    inputs3 = Input((img_rows, img_cols, img_depth),name="dyn3")
    inputs4 = Input((img_rows, img_cols, img_depth),name="dyn4")
    
    conv_out_1=single_unet(inputs1)
    conv_out_2=single_unet(inputs2)
    conv_out_3=single_unet(inputs3)
    conv_out_4=single_unet(inputs4)
    
    time_series=concatenate([conv_out_1, conv_out_2,conv_out_3,conv_out_4], axis=4)
    time_series=Lambda(backend_transpose)(time_series)
    
    CLSTM1=ConvLSTM2D(nb_filter=32, nb_row=5, nb_col=5, border_mode='same', return_sequences=True,dropout=0.2, recurrent_dropout=0.2)(time_series)
    CLSTM1=ConvLSTM2D(nb_filter=32, nb_row=5, nb_col=5, border_mode='same', return_sequences=True,dropout=0.2, recurrent_dropout=0.2)(CLSTM1)
    CLSTM1=TimeDistributed(Dropout(0.2))(CLSTM1)
    pool_CLSTM1=TimeDistributed(MaxPooling2D(pool_size=(2,2),padding='same'))(CLSTM1)
    CLSTM2=ConvLSTM2D(nb_filter=32, nb_row=5, nb_col=5, border_mode='same', return_sequences=True,dropout=0.2, recurrent_dropout=0.2)(pool_CLSTM1)
    CLSTM2=ConvLSTM2D(nb_filter=32, nb_row=5, nb_col=5, border_mode='same', return_sequences=False,dropout=0.2, recurrent_dropout=0.2)(CLSTM2)
    CLSTM2=TimeDistributed(Dropout(0.2))(CLSTM2)
    
    CLSTM3=Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(CLSTM2)
    CLSTM3 = BatchNormalization()(CLSTM3)
    CLSTM4 = Conv2D(16, (3, 3), activation='relu',padding='same')(CLSTM3)
    CLSTM4 = BatchNormalization()(CLSTM4)
    CLSTM4 = Conv2D(1, (1, 1), activation='sigmoid')(CLSTM4)
    
    model = Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=[CLSTM4])
    
    model.summary()
    return model

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 50.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate

def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train_dyn1=np.load('./0824/patch_dyn1_list_train.npy')
    imgs_train_dyn2=np.load('./0824/patch_dyn2_list_train.npy')
    imgs_train_dyn3=np.load('./0824/patch_dyn3_list_train.npy')
    imgs_train_dyn4=np.load('./0824/patch_dyn4_list_train.npy')

    imgs_train_dyn1=imgs_train_dyn1.transpose(0,2,3,1)
    imgs_train_dyn2=imgs_train_dyn2.transpose(0,2,3,1)
    imgs_train_dyn3=imgs_train_dyn3.transpose(0,2,3,1)
    imgs_train_dyn4=imgs_train_dyn4.transpose(0,2,3,1)
    
    imgs_mask_train=np.load('./0824/patch_mask_list_train.npy')
    imgs_mask_train=np.array(imgs_mask_train)
    imgs_mask_train=imgs_mask_train[:,3:-3,3:-3]
    imgs_mask_train=imgs_mask_train[:,:,:,np.newaxis]
    
    imgs_dev_dyn1=np.load('./0824/patch_dyn1_list_val.npy')
    imgs_dev_dyn2=np.load('./0824/patch_dyn2_list_val.npy')
    imgs_dev_dyn3=np.load('./0824/patch_dyn3_list_val.npy')
    imgs_dev_dyn4=np.load('./0824/patch_dyn4_list_val.npy')

    imgs_dev_dyn1=imgs_dev_dyn1.transpose(0,2,3,1)
    imgs_dev_dyn2=imgs_dev_dyn2.transpose(0,2,3,1)
    imgs_dev_dyn3=imgs_dev_dyn3.transpose(0,2,3,1)
    imgs_dev_dyn4=imgs_dev_dyn4.transpose(0,2,3,1)
    
    
    imgs_mask_dev=np.load('./0824/patch_mask_list_val.npy')
    imgs_mask_dev=np.array(imgs_mask_dev)
    imgs_mask_dev=imgs_mask_dev[:,3:-3,3:-3]
    imgs_mask_dev=imgs_mask_dev[:,:,:,np.newaxis]



    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = conv_clstm()
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    learning_rate = 0.001
    model.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00000199), loss=dice_coef_loss, metrics=['accuracy'])
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + 'model_{epoch:03d}.h5'), monitor='val_loss')

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + '.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit({"dyn1": imgs_train_dyn1, "dyn2": imgs_train_dyn2, "dyn3": imgs_train_dyn3, "dyn4": imgs_train_dyn4}, imgs_mask_train, batch_size=4, epochs=200, verbose=2, shuffle=True, validation_data=({"dyn1": imgs_dev_dyn1, "dyn2": imgs_dev_dyn2, "dyn3": imgs_dev_dyn3, "dyn4": imgs_dev_dyn4},imgs_mask_dev), callbacks=[model_checkpoint, csv_logger,lrate])


    print('-'*30)
    print('Training finished')
    print('-'*30)
    
def predict():
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    
    imgs_dev_dyn1=np.load('./0824/patch_dyn1_list_test.npy')
    imgs_dev_dyn2=np.load('./0824/patch_dyn2_list_test.npy')
    imgs_dev_dyn3=np.load('./0824/patch_dyn3_list_test.npy')
    imgs_dev_dyn4=np.load('./0824/patch_dyn4_list_test.npy')

    imgs_dev_dyn1=imgs_dev_dyn1.transpose(0,2,3,1)
    imgs_dev_dyn2=imgs_dev_dyn2.transpose(0,2,3,1)
    imgs_dev_dyn3=imgs_dev_dyn3.transpose(0,2,3,1)
    imgs_dev_dyn4=imgs_dev_dyn4.transpose(0,2,3,1)


    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model = conv_clstm()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)

    imgs_mask_test = model.predict({"dyn1": imgs_dev_dyn1, "dyn2": imgs_dev_dyn2, "dyn3": imgs_dev_dyn3, "dyn4": imgs_dev_dyn4}, batch_size=1, verbose=1)
    print(imgs_mask_test.shape)

    npy_mask_dir = 'test_mask_npy_new_0824'
    if not os.path.exists(npy_mask_dir):
        os.mkdir(npy_mask_dir)

    np.save(os.path.join(npy_mask_dir, project_name + '_tumor_mask.npy'), imgs_mask_test)
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    
if __name__ == '__main__':
    train()
    predict()