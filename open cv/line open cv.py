# -*- coding: utf-8 -*-
"""
drawing line
"""
#
import numpy as np 
import cv2
## Create a black image 
img = np.zeros((512,880,3), np.uint8)
img2 = np.zeros((512,880,3), np.uint8)
## Draw a diagonal blue line with thickness of 5 px 
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
img2 = cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imshow('sd',img)
cv2.imshow('sd',img2)
cv2.waitKey(0)
cv2.waitkey(0)


'''
drawing rectangle

'''
import numpy as np 
import cv2
img = np.zeros((512,880,3), np.uint8) 
img = cv2.rectangle(img,(311,311),(611,511),(0,255,0),6)
img=cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imshow('sd',img)
cv2.waitKey(0)

#
'''
drawing circle
'''
import numpy as np 
import cv2
## Create a black image 
img = np.zeros((512,512,3), np.uint8)
## Draw a diagonal blue line with thickness of 5 px 
img = cv2.circle(img,(511,511),345,(255,0,0),5)
cv2.imshow('sd',img)
cv2.waitKey(0)


'''
drawing elipse
'''
import numpy as np 
import cv2
img = np.zeros((512,512,3), np.uint8)
img = cv2.ellipse(img,(256,256),(100,50),180,180,180,255,-1)
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
cv2.imshow('sd',img)
cv2.waitKey(0)

'''
drawing polygon
'''
import numpy as np 
import cv2
img = np.zeros((512,512,3), np.uint8)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32) 
pts = pts.reshape((-1,1,2)) 
img = cv2.polylines(img,[pts],True,(0,255,255))
font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(img,'ADITYA',(100,500), font, 4,(0,255,0),2,cv2.LINE_AA)
cv2.imshow('sd',img)
cv2.waitKey(0)















