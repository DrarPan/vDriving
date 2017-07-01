#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2 as cv
import std_msgs.msg as smsg
import geometry_msgs.msg as gmsg
from sensor_msgs.msg import Image
import numpy as np

from cv_bridge import CvBridge, CvBridgeError

imageheight=480
imagewidth=640
imagechannel=3
class showImagewithVelocity:
	def __init__(self):
		self.image_sub=rospy.Subscriber("/camera/rgb/image_raw",Image,self.image_callback)
		self.cmd_vel_sub=rospy.Subscriber("/cmd_vel",gmsg.Twist,self.cmd_vel_callback)
		self.bridge=CvBridge()
		self.srcimg=np.zeros((imageheight,imagewidth,imagechannel))
		self.showimg=np.zeros((imageheight,imagewidth,imagechannel))
		self.twist=gmsg.Twist()
		self.vw=cv.VideoWriter("./vDriveVideo15.avi",cv.cv.CV_FOURCC('M','J','P','G'),15.0,(imagewidth,imageheight),True)#in OpenCv3: cv.VideoWriter_fourcc()
		print(self.vw.isOpened()," Waiting for Image Topic");

	def image_callback(self,data):
		try:
			self.srcimg=self.bridge.imgmsg_to_cv2(data,'bgr8')
		except CvBridgeError as e:
			print(e)
		cv.putText(self.srcimg,"Linear  : %.4f"%(self.twist.linear.x),(340,25),cv.FONT_HERSHEY_COMPLEX_SMALL,1.3,(70,20,250),2)
		cv.putText(self.srcimg,"Angular : %.4f"%(self.twist.angular.z),(330,50),cv.FONT_HERSHEY_COMPLEX_SMALL,1.3,(70,20,250),2)
		cv.imshow('image',self.srcimg)
		cv.waitKey(10)
		self.vw.write(self.srcimg)


	def cmd_vel_callback(self,data):
		self.twist=data

	def __del__(self):
		self.vw.release()

if __name__=='__main__':
	si=showImagewithVelocity()
	rospy.init_node('drive',anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")


