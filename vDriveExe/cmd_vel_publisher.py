#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2 as cv
import std_msgs.msg as smsg
import geometry_msgs.msg as gmsg
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf

batchsize=1
imageheight=120
imagewidth=160
imagechannel=3

linear_x_scale=0.5
angular_z_scale=1
max_linear_x=0.5
min_linear_x=-0.1
min_angular_z=-1
max_angular_z=1

vel_scale=0.4 #in test model, we should use slower speed

graph_lx=tf.Graph()
graph_az=tf.Graph()

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
	
	tf.train.Saver().restore(sess_az,'./model/model_linear_3conv_11_5_3_angular_z_30000')

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
	
	tf.train.Saver().restore(sess_lx,'./model/model_linear_3conv_11_5_3_linear_x10000')

class end2end_driving:
	def __init__(self):
		self.image_sub=rospy.Subscriber("/camera/rgb/image_raw",Image,self.image_callback,queue_size=1)
		self.cmd_vel_pub=rospy.Publisher("/cmd_vel",gmsg.Twist,queue_size=1)
		self.bridge=CvBridge()
		self.predict_lx=0
		self.predict_az=0
		self.twist=gmsg.Twist()
		self.shownwindow='Image'
		self.showimage= True

	def image_callback(self,data):
		try:
			cv_image=self.bridge.imgmsg_to_cv2(data)
		except CvBridgeError as e:
			print(e)
		cv_image=cv.resize(cv_image,(160,120))/255.0
		if self.showimage:
			cv.imshow(self.shownwindow,cv_image)
			cv.waitKey(10)
		with graph_az.as_default():
			self.predict_az=sess_az.run(predict_az,feed_dict={ImageInput_az: [cv_image],keep_prob_az: 1.0})
		with graph_lx.as_default():
			self.predict_lx=sess_lx.run(predict_lx,feed_dict={ImageInput_lx: [cv_image],keep_prob_lx: 1.0})
		print('Linear speed: %f, angular velocity: %f'%(self.predict_lx,self.predict_az))

		self.twist.linear.x=self.predict_lx*vel_scale*linear_x_scale
		self.twist.angular.z=self.predict_az*vel_scale*angular_z_scale

		if(self.twist.linear.x>max_linear_x*vel_scale):
			self.twist.linear.x=max_linear_x*vel_scale
		if(self.twist.linear.x<min_linear_x*vel_scale):
			self.twist.linear.x=min_linear_x*vel_scale
		if(self.twist.angular.z>max_angular_z*vel_scale):
			self.twist.angular.z=max_angular_z*vel_scale
		if(self.twist.angular.z<min_angular_z*vel_scale):
			self.twist.angular.z=min_angular_z*vel_scale

		self.cmd_vel_pub.publish(self.twist)

if __name__=='__main__':
	e2ed=end2end_driving()
	rospy.init_node('drive',anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

