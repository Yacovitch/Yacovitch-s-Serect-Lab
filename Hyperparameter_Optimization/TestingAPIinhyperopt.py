#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:26:44 2017

@author: jacob
"""

from __future__ import print_function
import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, concatenate,Dense, Activation
from keras.optimizers import Adadelta, Adam, rmsprop,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

from skimage.segmentation import mark_boundaries,find_boundaries
from os import makedirs

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperas import optim
from hyperas.distributions import uniform
from sklearn.metrics import roc_auc_score
import sys

N = '7.18.2017'

#def data():
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train#.astype('float32')
X_test = X_test#.astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#X_train = X_train / 255.0
#X_test = X_test / 255.0
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(X_test, y_test, batch_size=32)


space = {   'choice': hp.choice('Batch Norm',
                                [{'layers':'True1', }, #after every Conv layer, there is a Batch Norm
                                 {'layers':'True2'},   #one Batch Norm per Block
                                 {'layers':'False'},   #No Batch Norm
                      #           #'dropout4': hp.uniform('dropout4', .25,.50)}
                      #           {'layers':'four'},
                                ]),


            #'units1': hp.choice('units1', np.arange(10, 5000, dtype=int)),
            #'units2': hp.choice('units2', np.arange(64, 512, dtype=int)),

            #'dropout1': hp.uniform('dropout1', .25,.75),
            #'dropout3': hp.uniform('dropout3', .25,.75),
            #'dropout4': hp.uniform('dropout4', .25,.75),
            #'dropout5': hp.uniform('dropout5',  .25,.75),
            #'dropout6': hp.uniform('dropout6',  .25,.75),

            #'batch_size' : hp.choice('batch_size', np.arange(100, 200, dtype=int)),

            #'nb_epochs' :  hp.choice('nb_epochs', np.arange(3,50, dtype=int)),
            #'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            #'lrn' : hp.uniform('lrn',  .001,.0001),
            #'activation': 'relu',
            #'regularizer': hp.choice('regularizer',['l2','l1','l1_l2'])
        }

def model(params):
    print('Params testing: ', params)
    inputs = Input(shape=(3, 32, 32))
    
    if params['choice']['layers'] == 'True1':
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
    if params['choice']['layers'] == 'True2':
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
    if params['choice']['layers'] == 'False':
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    
    
    model.fit_generator(
    train_generator,
    samples_per_epoch=50000,
    nb_epoch=50,
    validation_data=validation_generator,
    validation_steps=10000 // 32,
    verbose=0)

    
    score = model.evaluate_generator(train_generator, steps = 10000 // 32)
    acc = score[1]
    
    #pred_auc = model.predict_proba(X_val, batch_size = 128, verbose = 0)
    #acc = roc_auc_score(Y_val, pred_auc)
    
    print('Accuracy:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(model, space, algo=tpe.suggest, max_evals=20
            , trials=trials)

print ('best: ')
print (best)
