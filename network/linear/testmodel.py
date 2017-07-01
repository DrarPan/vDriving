#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import tensorflow as tf

# design Networks
ImageInput=tf.placeholder(tf.float32,shape=[batchsize,120,160,3])
VelInput=tf.placeholder(tf.float32,shape=[batchsize,1])
kernel1=tf.Variable(tf.truncated_normal([7,7,3,64],dtype=tf.float32,stddev=0.01),name='weights1')
biases1=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),name='biases1')
conv1=tf.nn.relu(tf.nn.conv2d(ImageInput,kernel1,[1,2,2,1],padding="SAME")+biases1)
pool1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool1")

kernel2=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.01),name='weights2')
biases2=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[192]),name='biases2')
conv2=tf.nn.relu(tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")+biases2)
pool2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool2")

kernel3=tf.Variable(tf.truncated_normal([3,3,192,128],dtype=tf.float32,stddev=0.01),name='weights3')
biases3=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[128]),name='biases3')
conv3=tf.nn.relu(tf.nn.conv2d(pool2,kernel3,[1,1,1,1],padding="SAME")+biases3)

reshape=tf.reshape(conv3,[batchsize,-1])
dim=reshape.get_shape()[1].value;
weight_fc=tf.Variable(tf.truncated_normal([dim,1024],dtype=tf.float32,stddev=0.1),name="fc_weight")
bias_fc=tf.Variable(tf.constant(0.0,shape=[1024],dtype=tf.float32),name="fc_bias")
fc=tf.nn.relu(tf.matmul(reshape,weight_fc)+bias_fc,name="fc")

keep_prob=tf.placeholder(tf.float32)
fc_drop=tf.nn.dropout(fc,keep_prob)

weight_predict=tf.Variable(tf.truncated_normal([1024,1],dtype=tf.float32,stddev=0.1),name="predict_weight")
bias_predict=tf.Variable(tf.truncated_normal([1],dtype=tf.float32,stddev=0.1),name="predict_bias")
predict=tf.nn.bias_add(tf.matmul(fc_drop,weight_predict),bias_predict)

with tf.Session() as sess:
	saver=tf.train.Saver()
	sess=tf.Session()
	sess.run(tf.global_variables_initializer)