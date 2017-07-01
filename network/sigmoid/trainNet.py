#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import tensorflow as tf
import random
import loadData

shuffle=True
max_cmd_vel_linear_x=0.5
max_cmd_vel_angular_z=1.0

networktype='simple'
# first make data
inputs,labels=loadData.loadData('./data/data518/')
Ndata=len(labels)

images=[]
cmdvels=[]

randomidx=range(Ndata)
random.shuffle(randomidx)

for i in range(Ndata):
	images.append(inputs[randomidx[i]])
	cmdvels.append(labels[randomidx[i]])

for i in range(Ndata):
	cmdvels[i][0]/=max_cmd_vel_linear_x
	cmdvels[i][1]/=max_cmd_vel_angular_z

del inputs
del labels

# design Networks

batchsize=1;
if networktype=='5-layer-CNN':
	ImageInput=tf.placeholder(tf.float32,shape=[batchsize,160,120,3])
	VelInput=tf.placeholder(tf.float32,shape=[batchsize,2])
	kernel1=tf.Variable(tf.truncated_normal([7,7,3,64],dtype=tf.float32,stddev=0.1),name='weights1')
	biases1=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),name='biases1')
	conv1=tf.nn.tanh(tf.nn.conv2d(ImageInput,kernel1,[1,2,2,1],padding="SAME")+biases1)
	pool1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool1")
	dropout1=tf.nn.dropout(pool1,0.5)

	kernel2=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weights2')
	biases2=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[192]),name='biases2')
	conv2=tf.nn.relu(tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")+biases2)
	pool2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool2")
	dropout2=tf.nn.dropout(pool2,0.5)
	kernel3=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1),name='weights3')
	biases3=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[384]),name='biases3')
	conv3=tf.nn.relu(tf.nn.conv2d(pool2,kernel3,[1,1,1,1],padding="SAME")+biases3)

	kernel4=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=0.1),name='weights4')
	biases4=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name='biases4')
	conv4=tf.nn.relu(tf.nn.conv2d(conv3,kernel4,[1,1,1,1],padding="SAME")+biases4)

	kernel5=tf.Variable(tf.truncated_normal([3,3,256,128],dtype=tf.float32,stddev=0.1),name='weights5')
	biases5=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[128]),name='biases5')
	conv5=tf.nn.relu(tf.nn.conv2d(conv4,kernel5,[1,1,1,1],padding="SAME")+biases5)
	pool5=tf.nn.max_pool(conv5,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool5")

	reshape=tf.reshape(dropout2,[batchsize,-1])
	dim=reshape.get_shape()[1].value;
	weight_fc=tf.Variable(tf.truncated_normal([dim,1024],dtype=tf.float32,stddev=0.1),name="fc_weight")
	bias_fc=tf.Variable(tf.constant(0.0,shape=[1024],dtype=tf.float32),name="fc_bias")
	fc=tf.nn.relu(tf.matmul(reshape,weight_fc)+bias_fc,name="fc")

	weight_predict=tf.Variable(tf.truncated_normal([1024,2],dtype=tf.float32,stddev=0.1),name="predict_weight")
	bias_predict=tf.Variable(tf.truncated_normal([2],dtype=tf.float32,stddev=0.1),name="predict_bias")
	predict=tf.nn.tanh(tf.matmul(fc,weight_predict)+bias_predict,name="predict")
else if networktype='simple'
# ImageInput=tf.placeholder(tf.float32,shape=[batchsize,8])
# VelInput=tf.placeholder(tf.float32,shape=[batchsize,1])

# weight=tf.Variable(tf.truncated_normal([8,1],dtype=tf.float32,stddev=0.1))
# bias=tf.Variable(tf.constant(0.0,shape=[1],dtype=tf.float32))
# predict=tf.nn.tanh(tf.matmul(ImageInput,weight)+bias)

cost=0.5*tf.reduce_mean(tf.pow(tf.subtract(predict,VelInput),2.0))
optimizer=tf.train.AdamOptimizer(100)
train_step=optimizer.minimize(cost)

noiseImage=np.random.rand(batchsize,160,120,3)
noiseVel=np.random.rand(batchsize,2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(noiseVel)
	print("-----")
	for i in range(100000):
		sess.run(train_step,feed_dict={ImageInput: noiseImage, VelInput: noiseVel})
		print(sess.run(predict,feed_dict={ImageInput: noiseImage, VelInput: noiseVel}))
	print("-----")
	print(noiseVel)

# with tf.Session() as sess:
# 	noiseImage=np.random.rand(batchsize,160,120,3)
# 	noiseVel=np.random.rand(batchsize,2)
# 	print(noiseVel)
# 	sess.run(tf.global_variables_initializer())
# 	print(sess.run(cost,feed_dict={ImageInput: noiseImage, VelInput: noiseVel}))
# 	#print(noiseVel)
# 	for i in range(100):
# 		sess.run(train_step,feed_dict={ImageInput: noiseImage, VelInput: noiseVel})
# 		print(sess.run(predict,feed_dict={ImageInput: noiseImage, VelInput: noiseVel}))
# 		print(sess.run(cost,feed_dict={ImageInput: noiseImage, VelInput: noiseVel}))
