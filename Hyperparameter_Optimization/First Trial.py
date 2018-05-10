#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:28:33 2017

@author: jacob
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
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, concatenate,Dense, Activation, add
from keras.optimizers import Adadelta, Adam, rmsprop, Adamax, SGD
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.callbacks import ModelCheckpoint
from RocCallback import roc_callback
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

weight_name = 'C:/Users/Jacob/Desktop/Deep Learning/Results/TPE/weights.{epoch:01d}.hdf5'

#def data():
X = np.load('C:/Users/Jacob/Desktop/Pipline Project/Rearranging/Processed Data/DeepLearning/For Training/IMG_Train.npy')
X_val=np.load('C:/Users/Jacob/Desktop/Pipline Project/Rearranging/Processed Data/DeepLearning/For Training/IMG_Val.npy')
X = np.float64(X[:,:,48:96,48:96])
X_val = np.float64(X_val[:,:,48:96,48:96])

Y=np.load('C:/Users/Jacob/Desktop/Pipline Project/Rearranging/Processed Data/DeepLearning/For Training/Label_Train.npy')
Y = Y[:,0].astype(int)
Y_val=np.load('C:/Users/Jacob/Desktop/Pipline Project/Rearranging/Processed Data/DeepLearning/For Training/Label_Validation.npy')
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


            #'units1': hp.choice('units1', np.arange(64, 512, dtype=int)),
            #'units2': hp.choice('units2', np.arange(64, 512, dtype=int)),

            #'dropout1': hp.uniform('dropout1', 0.00,1.00),
            #'dropout3': hp.uniform('dropout3', .25,.75),
            #'dropout4': hp.uniform('dropout4', .25,.75),
            #'dropout5': hp.uniform('dropout5',  .25,.75),
            #'dropout6': hp.uniform('dropout6',  .25,.75),

            #'batch_size' : hp.choice('batch_size', np.arange(1, 200, dtype=int)),

            #'nb_epochs' :  hp.choice('nb_epochs', np.arange(3,50, dtype=int)),
            #'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'lrn' : hp.uniform('lrn',  1e-5,1e-2),
            #'activation': 'relu',
            #'regularizer': hp.choice('regularizer',['l2','l1','l1_l2'])
            #'rgl' : hp.uniform('rgl', 0,1.00)
        }

def model(params):
    print('Params testing: ', params)
    inputs = Input(shape = (6, 48,48))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    #x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    #x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    flat1 = Flatten(name='flatten')(x)
    
    dense1 = Dense(4096,activation = 'relu')(flat1)
    dense2 = Dense(4096,activation = 'relu')(dense1)
    output = Dense(2, activation='softmax', name='predictions')(dense2)
    
    
    
    model = Model(inputs=inputs, outputs=output)
    
    AUC_callback = roc_callback()
    
    model_checkpoint = ModelCheckpoint(weight_name, monitor='val_loss')
    
    model.compile(optimizer= SGD(lr=params['lrn'], momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    
    
    model.fit(X, Y, 
              batch_size=6, 
              validation_data=(X_val,Y_val),
              callbacks=[model_checkpoint, AUC_callback],
              epochs=40, verbose=1, shuffle=False)
    
    print(AUC_callback.aucs)
    best_auc = np.argmax(AUC_callback.aucs)
    print(best_auc)
    
    model.load_weights('C:/Users/Jacob/Desktop/Deep Learning/Results/TPE/weights.'+str(best_auc)+'.hdf5')
    
    Y_pred = model.predict(X_val, batch_size = 6, verbose = 1)
    
    acc = roc_auc_score(Y_val, Y_pred)
    
    print('AUC:', acc)
    
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(model, space, algo=tpe.suggest, max_evals=50, trials=trials)

print ('best: ')
print (best)