#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2 as cv
import time
import tensorflow as tf

batchsize=1
imageheight=120
imagewidth=160
imagechannel=3

# #Linear Speed

max_cmd_vel_linear_x=0.5
max_cmd_vel_angular_z=1

if __name__=='__main__':
	graph_lx=tf.Graph()
	graph_az=tf.Graph()

	imagefolder="../data/data518/"
	image=cv.imread(imagefolder+'image2483_rgb.png')/255.0

	pred_az=0;
	pred_lx=0;
	start=time.clock()
	with graph_az.as_default():
		sess_az=tf.Session()
		ImageInput_az=tf.placeholder(shape=[batchsize,imageheight,imagewidth,imagechannel],dtype=tf.float32)
		#Angular Velocity
		kernel1_az=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weights1_az')
		biases1_az=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),name='biases1_az')
		conv1_az=tf.nn.relu(tf.nn.conv2d(ImageInput_az,kernel1_az,[1,2,2,1],padding="SAME")+biases1_az,name="conv1_az")
		pool1_az=tf.nn.max_pool(conv1_az,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool1_az")

		kernel2_az=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weights2_az')
		biases2_az=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[192]),name='biases2_az')
		conv2_az=tf.nn.relu(tf.nn.conv2d(pool1_az,kernel2_az,[1,1,1,1],padding="SAME")+biases2_az,name="conv1_az")
		pool2_az=tf.nn.max_pool(conv2_az,[1,2,2,1],[1,2,2,1],padding='VALID',name="pool2_az")

		kernel3_az=tf.Variable(tf.truncated_normal([3,3,192,128],dtype=tf.float32,stddev=0.1),name='weights3_az')
		biases3_az=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[128]),name='biases3_az')
		conv3_az=tf.nn.relu(tf.nn.conv2d(pool2_az,kernel3_az,[1,1,1,1],padding="SAME")+biases3_az,name="conv1_az")

		reshape_az=tf.reshape(conv3_az,[batchsize,-1])
		dim=reshape_az.get_shape()[1].value;
		weight_fc_az=tf.Variable(tf.truncated_normal([dim,1024],dtype=tf.float32,stddev=0.1),name="fc_weight_az")
		bias_fc_az=tf.Variable(tf.constant(0.0,shape=[1024],dtype=tf.float32),name="fc_bias_az")
		fc_az=tf.nn.relu(tf.matmul(reshape_az,weight_fc_az)+bias_fc_az,name="fc_az")

		keep_prob_az=tf.placeholder(tf.float32)
		fc_drop_az=tf.nn.dropout(fc_az,keep_prob_az)

		weight_predict_az=tf.Variable(tf.truncated_normal([1024,1],dtype=tf.float32,stddev=0.1),name="predict_weight_az")
		bias_predict_az=tf.Variable(tf.truncated_normal([1],dtype=tf.float32,stddev=0.1),name="predict_bias_az")
		predict_az=tf.nn.bias_add(tf.matmul(fc_drop_az,weight_predict_az),bias_predict_az,name='predict_az')
		
		tf.train.Saver().restore(sess_az,'../model/model_linear_3conv_11_5_3_angular_z_30000')
	elapsed=(time.clock()-start)
	print("Time used for loading: ",elapsed)
	start=time.clock()

	with graph_lx.as_default():
		sess_lx=tf.Session()
		ImageInput_lx=tf.placeholder(shape=[batchsize,imageheight,imagewidth,imagechannel],dtype=tf.float32)
		#Linear Speed
		kernel1_lx=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weights1_lx')
		biases1_lx=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),name='biases1_lx')
		conv1_lx=tf.nn.relu(tf.nn.conv2d(ImageInput_lx,kernel1_lx,[1,2,2,1],padding="SAME")+biases1_lx,name="conv1_lx")
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
		
		tf.train.Saver().restore(sess_lx,'../model/model_linear_3conv_11_5_3_linear_x10000')
	elapsed=(time.clock()-start)
	print("Time used for loading: ",elapsed)

	start=time.clock()
	with graph_az.as_default():
		pred_az=sess_az.run(predict_az,feed_dict={ImageInput_az: [image],keep_prob_az: 1.0})
	with graph_lx.as_default():
		pred_lx=sess_lx.run(predict_lx,feed_dict={ImageInput_lx: [image],keep_prob_lx: 1.0})

	elapsed=(time.clock()-start)
	print("Time used for loading: ",elapsed)
	
	print('Linear speed: %f, angular velocity: %f'%(pred_lx,pred_az))
	cv.imshow('image',image)
	cv.waitKey(5000)