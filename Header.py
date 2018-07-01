# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 13:59:16 2018

@author: Dell
"""


'''
This is the header of the program.
In this we are calculating accuracy of the model.
We can also predict the data by reading a data and then predicting it.

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



ROOT_PATH = "E:\python program\CUB_200_2011"
train_data_directory = os.path.join(ROOT_PATH, "images")
test_data_directory = os.path.join(ROOT_PATH, "images")

images, labels = load_data(train_data_directory)

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]


#a = [np.zeros((224,224,3)), np.zeros((224,224,3)), np.zeros((10,224,3))]
#np.array(a)

# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')
        
# Load the test data
test_images, test_labels = load_test_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
from skimage.color import rgb2gray

test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

#This to predict the name of the bird by readig the image from

test_images28 = transform.resize(image, (28, 28))

# Convert to grayscale
from skimage.color import rgb2gray

test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict = test_images28)[0]
