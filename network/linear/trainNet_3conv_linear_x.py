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
bias_cmd_vel_angular=0
max_iteration=10001;
disp_epoch=50
save_epoch=2000
#imgwidth=80;
#imgheight=60;

# first make data
inputs,labels=makeData.loadData('../data/data518/',resize=None)
batchsize=16;
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
kernel1_lx=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weights1_lx')
biases1_lx=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),name='biases1_lx')
conv1_lx=tf.nn.relu(tf.nn.conv2d(ImageInput,kernel1_lx,[1,2,2,1],padding="SAME")+biases1_lx,name="conv1_lx")
pool1_lx=tf.nn.max_pool(conv1_lx,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool1_lx")

kernel2_lx=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weights2_lx')
biases2_lx=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[192]),name='biases2_lx')
conv2_lx=tf.nn.relu(tf.nn.conv2d(pool1_lx,kernel2_lx,[1,1,1,1],padding="SAME")+biases2_lx,name="conv1_lx")
pool2_lx=tf.nn.max_pool(conv2_lx,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool2_lx")

kernel3_lx=tf.Variable(tf.truncated_normal([3,3,192,128],dtype=tf.float32,stddev=0.1),name='weights3_lx')
biases3_lx=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[128]),name='biases3_lx')
conv3_lx=tf.nn.relu(tf.nn.conv2d(pool2_lx,kernel3_lx,[1,1,1,1],padding="SAME")+biases3_lx,name="conv1_lx")

reshape_lx=tf.reshape(conv3_lx,[batchsize,-1])
dim=reshape_lx.get_shape()[1].value;
weight_fc_lx=tf.Variable(tf.truncated_normal([dim,1024],dtype=tf.float32,stddev=0.1),name="fc_weight_lx")
bias_fc_lx=tf.Variable(tf.constant(0.0,shape=[1024],dtype=tf.float32),name="fc_bias_lx")
fc_lx=tf.nn.relu(tf.matmul(reshape_lx,weight_fc_lx)+bias_fc_lx,name="fc_lx")

keep_prob_lx=tf.placeholder(tf.float32)
fc_drop_lx=tf.nn.dropout(fc_lx,keep_prob_lx)

weight_predict_lx=tf.Variable(tf.truncated_normal([1024,1],dtype=tf.float32,stddev=0.1),name="predict_weight_lx")
bias_predict_lx=tf.Variable(tf.truncated_normal([1],dtype=tf.float32,stddev=0.1),name="predict_bias_lx")
predict_lx=tf.nn.bias_add(tf.matmul(fc_drop_lx,weight_predict_lx),bias_predict_lx,name='predict_lx')

cost=0.5*tf.reduce_mean(tf.square(predict_lx-VelInput))
optimizer=tf.train.AdamOptimizer(1e-3)
#with tf.device('gpu:0'):
train_step=optimizer.minimize(cost)

with tf.Session() as sess:
	f=open('./performance/model_linear_3conv_11_5_3_linear_x.txt','w')
	c_bar=0
	saver=tf.train.Saver()
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
		c, p, _ =sess.run([cost,predict_lx,train_step],feed_dict={ImageInput: feedimage, VelInput: feedlabel, keep_prob_lx: 0.6})
		c_bar+=c
		
		#print("The ",i," iteration, cost: ",c)
		if i%disp_epoch==0 and i>0:
			f.write("%d: %f\n"%(i,c))
			print(i," iteration, average lost: ",c_bar/disp_epoch)
			print(np.column_stack((p,feedlabel)))
			c_bar=0

		if i%save_epoch==0 and i>0:
			save_path=saver.save(sess,"../model/model_linear_3conv_11_5_3_linear_x"+str(i))
			print("Model saved in file: %s" % save_path)
	f.close()
