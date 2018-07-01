# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 13:59:16 2018

@author: Dell
"""



'''

In this function we are training our data.
Every species of bird is contained in a seprate directory.
So i have used 70% of data of each species for training purpose.
The rest you will understand in the code given below.

'''

#importing tensorflow
import tensorflow as tf
import os
from skimage import data
import numpy as np
# Import the `transform` module from `skimage`
from skimage import transform
# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    print(len(directories))
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpg")]
        c = 0
        
        for f in file_names:
            if(0.7*len(file_names) >= c):
                images.append(data.imread(f))
                #print(f)
                #print("")
                k = d.split('.')[1]
                #print(k)
                labels.append(k)
            c = c + 1
        #break;
    return images, labels
