#!/usr/bin/env python
'''
Validation of face and person recognition running in Real Time.

'''
from __future__ import print_function

import numpy as np
import cv2
import os

sv = np.loadtxt('data/data_images/v.data',np.float32)
sw = np.loadtxt('data/data_images/w.data',np.float32)

# names=["Edu","Antonio","Jorge","Patty","JP","America","Amoran"]
names=os.listdir("data/data_images/Persons/")

def nothing(x):
    pass

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    width, height = rects.shape[:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[1]
    except:
        video_src = 0

    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow('facedetect')
    # create trackbars for color change
    
    while True:
        ret, img = cam.read()
        # img = cv2.imread('cara5.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = detect(gray, cascade)
        
        vis = img.copy()
        draw_rects(vis,rects,(0,255,0))
        
        num=0        
        for x1,y1,x2,y2 in rects:
            crop_img = img[y1:y2, x1:x2]
            #--------------------------------------------------------
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            #Basic threshold example
            #Set threshold and maxValue
            thresh=127
            maxValue=255
            th,dst=cv2.threshold(gray,thresh,maxValue,cv2.THRESH_BINARY);
            resized_image=cv2.resize(dst,(50,50))
            cv2.imshow('person%d'%num,resized_image)
            #############################################################################3
            #-----------------------------------------------------
            #RNA
            #-----------------------------------------------------
            resized_image=np.asarray(resized_image, dtype='uint8')
            h,w=resized_image.shape
            sample = resized_image.reshape((1,w*h))
            sample = sample/255 ##value to compare
            
            h,w=sample.shape
            bias=np.ones((h,1))
            x=np.column_stack((sample,bias))
            
            #Computing output
            inX=x.transpose()
            m=sv.transpose().dot(inX)
            n=1.0/(1.0+np.exp(-m)) #sigmond 1
            # n = 2.0/(1+np.exp(-m)) - 1  #sigmoid 2
            # n = np.exp(-np.asarray(m)*np.asarray(m));                 # Gaussian
            
            out = sw.transpose().dot(n)
            valMaxY=np.max(out)
            print(valMaxY)
            if (valMaxY>0.5):
                print(valMaxY)
                person=np.argmax(out)
                cv2.putText(vis,"%s"%names[person], (x1+10,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,250),2)
            #############################################################################
            num+=1
       
        cv2.imshow('facedetect', vis)
        
        key = cv2.waitKey(2)
        if key == 27:
            break
        
    cv2.destroyAllWindows()
