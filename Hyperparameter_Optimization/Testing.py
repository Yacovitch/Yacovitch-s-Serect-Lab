#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:18:39 2017

@author: jacob
"""
import numpy as np

X = np.load('/home/jacob/Deep Learning/cancer_classification/data/Augmented Data/labels_test.npy')[:,0].astype(int)

def binary_train(data):
    t = data
    x = np.zeros((t.shape[0],2),dtype = int)
    counter = [0,0]
    for i in range (0,t.shape[0]):
        z = np.argmax(t[i,:])
        if z > 0:            
            x[i,1] = 1           
            counter[0] +=1            
        else:            
            x[i,0] = 1            
            counter [1] +=1
    return x

Y = binary_train(X)