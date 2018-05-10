# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:34:49 2017

@author: Jacob
"""

from __future__ import print_function

import os

with open(os.path.expanduser('~')+'\\.keras\\keras.json','w') as f:
    new_settings = """{\r\n
    "epsilon": 1e-07,\r\n
    "image_dim_ordering": "th",\n
    "backend": "theano",\r\n
    "floatx": "float32"\r\n
    }"""
    f.write(new_settings)
    
from keras import backend as K
K.set_image_dim_ordering("th")

import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import ZeroPadding2D,Input, Convolution2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, concatenate,Dense, Activation, add, AveragePooling2D,GlobalAveragePooling2D
from keras.optimizers import Adadelta, Adam, rmsprop, Adamax, SGD
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1_l2, l2
from keras import backend as K
import matplotlib.pyplot as plt
from scipy import ndimage 

from skimage.segmentation import mark_boundaries,find_boundaries
from os import makedirs

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperas import optim
from hyperas.distributions import uniform
from sklearn.metrics import roc_auc_score
from resnet import conv_block, identity_block


#def data():
X = np.load('C:/Users/Jacob/Desktop/Deep Learning/Data/Augmented Data/Prostate Data/drive-download-20170831T175316Z-001/TRAIN_IMGv1_AUG1.npy')
X_val=np.load('C:/Users/Jacob/Desktop/Deep Learning/Data/UnAugmented Data/Prostate/drive-download-20170913T165057Z-001/TEST_IMGv1_780x2x144x144.npy')
X = X[:,:,16:128,16:128]
X_val = X_val[:,:,16:128,16:128]

Y=np.load('C:/Users/Jacob/Desktop/Deep Learning/Data/Augmented Data/Prostate Data/drive-download-20170831T175316Z-001/TRAIN_LBSv1_AUG1.npy')
Y = Y[:,0].astype(int)
Y_val=np.load('C:/Users/Jacob/Desktop/Deep Learning/Data/UnAugmented Data/Prostate/drive-download-20170913T165057Z-001/TEST_LBSv1_780x2x144x144.npy')
Y_val=Y_val[:,0].astype(int)

for i in range(0,X.shape[1]):
    M = X[:,i]
    mu = np.mean(M)
    std = np.std(M)
    X[:,i]-=mu
    X[:,i]/=std

for i in range(0,X_val.shape[1]):
    M = X_val[:,i]
    mu = np.mean(M)
    std = np.std(M)
    X_val[:,i]-=mu
    X_val[:,i]/=std
'''
t = Y
x = np.zeros((t.shape[0],2),dtype = int)
for i in range (0,t.shape[0]):
    z = np.argmax(t[i,:])
    if z > 0:            
        x[i,1] = 1
    else:            
        x[i,0] = 1
Y = x

ty = Y_val
xy = np.zeros((ty.shape[0],2),dtype = int)
for i in range (0,ty.shape[0]):
    zy = np.argmax(ty[i,:])
    if zy > 0:            
        xy[i,1] = 1
    else:            
        xy[i,0] = 1
Y_val = xy
'''

space = {   #'choice': hp.choice('num_layers',
                      #          [{'layers':'one', },
                      #           {'layers':'two'  ,'dropout2': hp.uniform('dropout2', .25,.60)},
                      #           {'layers':'three', 'dropout3': hp.uniform('dropout3', .25,.50), 
                      #           'dropout4': hp.uniform('dropout4', .25,.50)}]),


            'units1': hp.choice('units1', np.arange(0, 5, dtype=int)),
            'units2': hp.choice('units2', np.arange(0, 20, dtype=int)),
            'units3': hp.choice('units3', np.arange(0, 20, dtype=int)),

            #'dropout1': hp.uniform('dropout1', .25,.75),
            #'dropout3': hp.uniform('dropout3', .25,.75),
            #'dropout4': hp.uniform('dropout4', .25,.75),
            #'dropout5': hp.uniform('dropout5',  .25,.75),
            #'dropout6': hp.uniform('dropout6',  .25,.75),

            #'batch_size' : hp.choice('batch_size', np.arange(100, 200, dtype=int)),

            #'nb_epochs' :  hp.choice('nb_epochs', np.arange(3,50, dtype=int)),
            #'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'lrn' : hp.uniform('lrn',  1e-4,1e-2),
            #'activation': 'relu',
            #'regularizer': hp.choice('regularizer',['l2','l1','l1_l2'])
            #'rgl' : hp.uniform('rgl', 0,1.00)
        }

def model(params):
    print('Params testing: ', params)
    
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    #global bn_axis

    bn_axis = 1
    img_input = Input(shape=(2, 112, 112), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    #x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 2, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    for i in range(1,params['units1']):
        x = identity_block(x, 2, [32, 32, 128], stage=2, block='b'+str(i))

    
    x = conv_block(x, 2, [64, 64, 256], stage=3, block='a')
    for i in range(1,params['units2']):
      x = identity_block(x, 2, [64, 64, 256], stage=3, block='b'+str(i))

    x = conv_block(x, 2, [128, 128, 512], stage=4, block='a')
    for i in range(1,params['units3']):
      x = identity_block(x, 2, [128, 128, 512], stage=4, block='b'+str(i))
    '''
    x = conv_block(x, 2, [256, 256, 1024], stage=5, block='a')
    x = identity_block(x, 2, [256, 256, 1024], stage=5, block='b')
    x = identity_block(x, 2, [256, 256, 1024], stage=5, block='c')
    '''
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(2, activation='softmax', name='fc1000')(x_fc)

    
    model = Model(img_input, x_fc)
    
    model.compile(optimizer=SGD(lr=params['lrn'], decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

    
    model.fit(X, Y, 
              batch_size=6, 
              validation_data=(X_val,Y_val),
              epochs=2, verbose=0, shuffle=True)

    pred_auc = model.predict(X_val, batch_size = 6, verbose = 0)
    acc = roc_auc_score(Y_val, pred_auc)
    print('AUC:', acc)

    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(model, space, algo=tpe.suggest, max_evals=50, trials=trials)

print ('best: ')
print (best)