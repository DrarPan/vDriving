#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2 as cv

#np.set_printoptions(threshold='nan')
def loadData(directory,inputdatatype="rgb",labeldatatype="cmd_vel",resize=None,datarange=None):
	f=open(directory+'/data.txt','r')
	alldata=f.readlines()
	Ndata=len(alldata)//10
	imagedata=[]
	labeldata=[]
	
	for i in range(Ndata):
		label=int(alldata[i*10]);

		if datarange:
			if not (label in datarange):
				continue;

		if inputdatatype=='rgb':
			rgbimg=cv.imread(directory+'/image'+str(label)+'_rgb.png')
			rgbimg=rgbimg/255.0
			if resize:
				rgbimg=cv.resize(rgbimg,resize)
			imagedata.append(rgbimg);
		if inputdatatype=='gray':
			rgbimg=cv.imread(directory+'/image'+str(label)+'_rgb.png')
			rgbimg=cv.cvtColor(rgbimg,cv.COLOR_RGB2BGR)
			grayimg=cv.cvtColor(rgbimg,cv.COLOR_BGR2GRAY)/255.0
			if resize:
				grayimg=cv.resize(grayimg,resize)
			grayimg=grayimg.reshape(resize[1],resize[0],1)
			imagedata.append(grayimg)
		elif inputdatatype=='d':
			dimg=cv.imread(directory+'/image'+str(label)+'_depth.png')
			dimg=dimg/32.0
			if resize:
				dimg=cv.resize(dimg,resize)
			imagedata.append(dimg);
		elif inputdatatype=='rgbd':
			rgbimg=cv.imread(directory+'/image'+str(label)+'_rgb.png')
			dimg=cv.imread(directory+'/image'+str(label)+'_depth.png')[:,:,0:1]
			rgbimg=rgbimg/255.0
			dimg=dimg/32.0
			if resize:
				rgbimg=cv.resize(rgbimg,resize)
				dimg=cv.resize(dimg,resize)
			imagedata.append(np.concatenate((rgbimg,dimg),2));
		
		if labeldatatype=='cmd_vel':
			labeldata.append([float(alldata[10*i+2]),float(alldata[10*i+3])])
		elif labeldatatype=='vel':
			labeldata.append([float(alldata[10*i+7]),float(alldata[10*i+8])])
		elif labeldatatype=='pose':
			labeldata.append([float(alldata[10*i+4]),float(alldata[10*i+5]),float(alldata[10*i+6])])

	return imagedata, labeldata

def imageenhance(images,labels,intensitybiases,applyNoise=True,maxnoise=0.01,numnoise=3,truncate=True):
	enhanceimages=[]
	enhancelabels=[]
	imgshape=images[0].shape
	for i,image in enumerate(images):
		for intensitybias in intensitybiases:
			newimage=image+intensitybias	
			if truncate:
				newimage[newimage>1.0]=1.0
				newimage[newimage<0.0]=0.0
			enhanceimages.append(newimage)
			enhancelabels.append(labels[i])
			if applyNoise==True:
				for n in range(numnoise):
					noiseimage=newimage+(np.random.rand(imgshape[0],imgshape[1],imgshape[2])*maxnoise*2-maxnoise)
					if truncate:
						noiseimage[noiseimage>1.0]=1.0
						noiseimage[noiseimage<0.0]=0.0
					enhanceimages.append(noiseimage)
					enhancelabels.append(labels[i])
	return enhanceimages,enhancelabels;
#TODO: delete the point with speed (0.0,0.0)?

if __name__=="__main__":
	images,labels=loadData("./data/data518",inputdatatype='gray',resize=(160,120));
	images,labels=imageenhance(images,labels,[-0.1,0.0,0.1])

	# print(images[0].shape)
	# print(images[0])
	print(len(images))
	sp=images[0].shape

	for i in range(48,72):
		cv.imshow("i",images[i])
		cv.waitKey(2000)
		print(labels[i])

