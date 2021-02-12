# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:09:41 2019

@author: prateek jain
"""

import numpy as np
import cv2

img=cv2.imread('b1.png',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##############################################################################

#img=cv2.imread('b1.png',0)
#cv2.imshow('image',img)
#k=cv2.waitKey(0)
#
#if k==27:
#    cv2.destroyAllWindows()
#elif K==ord('s'):
#    cv2.imwrite('b1_gry.png',img)
#cv2.destroAllWindows()

##############################################################################

#cap=cv2.VideoCapture(0)
#
#while(True):
#    ret,frame=cap.read()
#    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    
#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#    
#cap.release()
#cv2.destroyAllWindows()

##############################################################################

#cap=cv2.VideoCapture(0)
#fourcc=cv2.VideoWriter_fourcc(*'XVID')
#out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
#while(cap.isOpend()):
#    ret,frame=cap.read()
#    if ret==True:
#        frame=cv2.flip(frame,0)
#        out.write(frame)
#        cv2.imshow('frame',frame)
#        if cv2.waitKey(1) & 0xFF==ord('q'):
#            break
#    else:
#        break
#
#cap.release()
#out.release()
#cv2.destroyAllWindows()
    
##############################################################################

#cap=cv2.VideoCapture(0)
#while(cap.isOpend()):
#    ret,frame=cap.read()
#    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()