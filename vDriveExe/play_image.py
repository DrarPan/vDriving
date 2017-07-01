#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

directory='../data/data518/'
f=open(directory+'/data.txt','r')
alldata=f.readlines()
Ndata=len(alldata)//10
imagedata=[]

imageheight=120
imagewidth=160
shownwindow="Image"

rospy.init_node('drive',anonymous=True)
rate=rospy.Rate(5)
bridge=CvBridge()
image_pub=rospy.Publisher('/camera/rgb/image_raw',Image,queue_size=1)

for i in range(Ndata):
	label=int(alldata[i*10])

	rgbimg=cv.imread(directory+'/image'+str(label)+'_rgb.png')
	rgbimg=cv.resize(rgbimg,(imagewidth,imageheight))
	cv.imshow(shownwindow,rgbimg)
	if(cv.waitKey(10)==99):
		break
	try:
		image_pub.publish(bridge.cv2_to_imgmsg(rgbimg,"bgr8"))
	except CvBridgeError as e:
		print(e)
	rate.sleep()
	

