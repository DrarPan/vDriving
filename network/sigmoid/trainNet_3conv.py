#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import tensorflow as tf
import random
import sys
sys.path.append('..')
import makeData

shuffling=True
scaling=True
max_cmd_vel_linear_x=0.5
max_cmd_vel_angular_z=1
bias_cmd_vel_angular=0.5
max_iteration=1000;
disp_epoch=10
save_epoch=200
#imgwidth=80;
#imgheight=60;

# first make data
inputs,labels=makeData.loadData('../data/data518/',resize=None)
batchsize=50;
Ndata=len(labels)

vmode=0 #0 for speed, 1 for angular velocity

images=[]
cmdvels=[]

randomidx=range(Ndata)
random.shuffle(randomidx)

for i in range(Ndata):
	images.append(inputs[randomidx[i]])
	cmdvels.append(labels[randomidx[i]])

for i in range(Ndata):
	cmdvels[i][0]=cmdvels[i][0]/max_cmd_vel_linear_x
	cmdvels[i][1]=cmdvels[i][1]/max_cmd_vel_angular_z+bias_cmd_vel_angular

#print(cmdvels)
cmdvels=np.array(cmdvels)

del inputs
del labels

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

weight_predict=tf.Variable(tf.truncated_normal([1024,1],dtype=tf.float32,stddev=0.1),name="predict_weight")
bias_predict=tf.Variable(tf.truncated_normal([1],dtype=tf.float32,stddev=0.1),name="predict_bias")
predict=tf.nn.bias_add(tf.matmul(fc,weight_predict),bias_predict)

cost=0.5*tf.reduce_mean(tf.square(predict-VelInput))
optimizer=tf.train.AdamOptimizer(1e-3)
train_step=optimizer.minimize(cost)

with tf.Session() as sess:
	c_bar=0
	sess.run(tf.global_variables_initializer())
	for i in range(max_iteration):
		startidx=(i*batchsize)%Ndata

		if startidx>Ndata-batchsize:
			feedimage=images[startidx:startidx+batchsize]+images[0:batchsize-(Ndata-startidx)]
			feedlabel=np.reshape(np.concatenate((cmdvels[startidx:startidx+batchsize,vmode],cmdvels[0:batchsize-(Ndata-startidx),vmode])),(-1,1))	
		else:
			feedimage=images[startidx:startidx+batchsize]
			feedlabel=np.reshape(cmdvels[startidx:startidx+batchsize,vmode],(-1,1))
		#print(np.reshape(cmdvels[startidx:startidx+batchsize,0],(-1,1)))
		c, p, _ =sess.run([cost,predict,train_step],feed_dict={ImageInput: feedimage, VelInput: feedlabel})
		c_bar+=c
		
		#print("The ",i," iteration, cost: ",c)

		if i%disp_epoch==0 and i>0:
			print(i," iteration, average cost: ",c_bar/disp_epoch)
			print(p)
			c_bar=0

		if i%save_epoch==0 and i>0:
			save_path=saver.save(sess,"../model/model417/model")
			print("Model saved in file: %s" % save_path)
			
