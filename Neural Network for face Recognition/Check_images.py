#!/usr/bin/env python
'''
[Extra]
Create data beging for image which already exist.

USAGE:
    Python Check_images.py or ./Check_images.py

Pres any button to go next.    
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

list=os.listdir("data/data_images/Persons/Edu")
if (len(list)==0):
    print("Empty folder")
    quit()


#---------------------------------------------------------------------------
# Uncomment in order to aks in terminal
name = raw_input("Who do you want to check? ")
# Fix a name and number of  pictures.
# name ="Edu"
#---------------------------------------------------------------------------
      
list=os.listdir("data/data_images/Persons/Edu")
n_images=len(list) #image
if (n_images==0):
    print("Empty folder")
    quit()



def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0

    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    #############################################################################3
    n=0;
    samples =  np.empty((0,2500))
    
    for i in range(0,int(n_images)):
        img = cv2.imread('data/data_images/Persons/%s/cara%d.jpg'%(name,i))
        # img = cv2.imread('cara1.jpg')
            
        if not (type(img) is np.ndarray): 
            print("There is no Image. Run first \"Take_photo.py\"")
            quit()

    #############################################################################3
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
        	crop_img = img[y1:y2, x1:x2]

        #--------------------------------------------------------
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        #Basic threshold example
        #Set threshold and maxValue
        thresh=127
        maxValue=255
        th,dst=cv2.threshold(gray,thresh,maxValue,cv2.THRESH_BINARY);

        resized_image = cv2.resize(dst, (50, 50)) 
        cv2.imshow(name, resized_image)

        #-----------------------------------------------------
        #RNA
        #-----------------------------------------------------
              
        # print ("training complete")
        # np.savetxt('generalsamples.data',samples)
        #############################################################################3
        resized_image=np.asarray(resized_image, dtype='uint8' )
        h,w=resized_image.shape
        sample = resized_image.reshape((1,w*h))
        sample = sample/255
        samples = np.append(samples,sample,0)
        n=n+1;
        
        #############################################################################3

        # cv2.imshow('facedetect', vis)
        cv2.waitKey(0)

    #############################################################################3
    np.savetxt('data/data_images/Persons/%s/generalsamples.data'%name,samples)
    #############################################################################3

    cv2.destroyAllWindows()
