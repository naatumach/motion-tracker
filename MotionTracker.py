# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:24:43 2021

@author: vsharma
https://gist.githubusercontent.com/pknowledge/623515e8ab35f1771ca2186630a13d14/raw/42fed016562b6b88db75d1ba9a265d8ffeb337ad/basic_motion_detection_opencv_python.py
https://github.com/misbah4064/object_tracking/blob/master/object_tracking.py
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
video = cv2.VideoCapture('vid.mp4')
https://www.youtube.com/watch?v=zHnZ7sVHy1s
"""
import cv2
import sys
#import imutils
import numpy as np

TrDict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'boosting': cv2.TrackerBoosting_create,
          'mil':cv2.TrackerMIL_create,
          'tld':cv2.TrackerTLD_create,
          'medianflow' : cv2.TrackerMedianFlow_create,
          'mosse':cv2.TrackerMOSSE_create}

trackers = cv2.MultiTracker_create()

#v = cv2.VideoCapture('box5.mp4')
#v = cv2.VideoCapture('vid3.mp4')
#v = cv2.VideoCapture('still.mp4')
v = cv2.VideoCapture(r'2021-01\vid12.mp4')


class ClassBox:
    
    def __init__(self,bufferArrayi=0):
        self.box_id=None
        self.bccAfter = None
        self.bccBefore = None
        self.dist = None

        self.bufferArrayX=np.zeros(50)
        self.bufferArrayY=np.zeros(50)
        self.bufferArrayi = bufferArrayi
        
    
    def setup (self,x,y,w,h):
        #self.box_id = i
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.bcc = self.boxCenterCoordCalc()
        
    def setID(self,i):
        self.box_id = i
        
    def boxCenterCoordCalc(self):
        self.boxCenterCoord = (self.x+(self.w/2), self.y+(self.h/2))
        self.boxTemp= [0,0]
        self.boxTemp[0]= int(self.boxCenterCoord[0])
        self.boxTemp[1]= int(self.boxCenterCoord[1])
        self.boxCenterCoord = np.array(self.boxTemp)
        return (self.boxCenterCoord)
    

    def bccSetPosition(self,frameNumber):
        self.bccAfter = self.boxCenterCoordCalc()
        if frameNumber == 3:
            self.bccBefore = self.boxCenterCoordCalc()
            
    
    def refresh(self):
        return
    
    def calcDist(self):    
        self.dist = np.linalg.norm(self.bccAfter - self.bccBefore)
        return (self.dist)
    
    def getDirection(self,frameNumber):
        self.direction = self.bccAfter - self.bccBefore
        self.directionAfterX = self.direction[0]
        self.directionAfterY = self.direction[1]
        if frameNumber == 3:
            self.directionBeforeX = self.direction[0]
            self.directionBeforeY = self.direction[1]
        return
    
    def printDir(self):
        print(self.box_id,self.directionBeforeY,self.directionAfterY)
    
    def bufferArrayReset(self):
        if self.bufferArrayi>len(self.bufferArrayX)-1:
            self.bufferArrayi = 0
        return

    def processBuffer(self):
        #print ("*************",self.box_id,"|",self.bufferArrayi)
        self.bufferArrayX[self.bufferArrayi] = self.directionAfterX
        self.bufferArrayY[self.bufferArrayi] = self.directionAfterY
        self.bufferArrayi = self.bufferArrayi+1
        self.bufferArrayReset()
        self.bufferArrayMeanX = np.mean(np.absolute(self.bufferArrayX))
        self.bufferArrayMeanY = np.mean(np.absolute(self.bufferArrayY))
        #print ("-------------",self.box_id,"|",self.bufferArrayi)
        

    def checkStopped(self):
        #print (self.bufferArrayMean)
        self.stoppedAxis = None
        self.stoppedY = False
        if self.bufferArrayMeanY == 0.0:
            self.stoppedY = True
            return True

        
    def checkStoppedXY(self):
        #print (self.bufferArrayMean)
        self.stoppedAxis = None
        self.stoppedX = False
        self.stoppedY = False
        if self.bufferArrayMeanX == 0.0:
            self.stoppedY = True
        if self.bufferArrayMeanX == 0.0:
            self.stoppedX = True
            
        if self.stoppedX == True and self.stoppedY == True:
            return "XY"
        
        if self.stoppedX == True:
            return "X"

        if self.stoppedY == True:
            return "Y"


    
    def setBefores(self):
        self.bccBefore = self.boxCenterCoordCalc()
        self.directionBeforeY = self.direction[1]
        



# = cv2.VideoCapture(0)

ret, frame = v.read()
#frame = imutils.resize(frame,width=600)

k = 3

roi = {boxlist: ClassBox() for boxlist in range(k)} #Region of interest

for i in range (k):
    cv2.imshow('Frame',frame)
    bbi=cv2.selectROI('Frame',frame)
    tracker_i = TrDict['kcf']()
    trackers.add(tracker_i,frame,bbi)

    
frameNumber = 2    
baseDir = r'Results'
#bufferArray=np.zeros(10)
#bufferArrayi = 0
while True:
    ret, frame = v.read()
    if not ret:
        break
    (success, boxes) = trackers.update(frame)
    #np.savetxt(baseDir+'/frame_'+str(frameNumber)+'.txt',boxes,fmt='%f')
    frameNumber += 1
    for i,box in enumerate(boxes):
#    if success:
        (x,y,w,h) = [int(a) for a in box]
        roi[i].setup(x,y,w,h)
        roi[i].setID(i)
        cv2.rectangle(frame,(roi[i].x,roi[i].y),(roi[i].x+roi[i].w,roi[i].y+roi[i].h),(100,225,0),2)
        cv2.circle(frame,(roi[i].bcc[0],roi[i].bcc[1]),1,(0,0,255),2)
        roi[i].bccSetPosition(frameNumber)
        roi[i].getDirection(frameNumber)
        roi[i].setBefores()
        
        #print ("i {} ,box_id {},buffer_arrayi {},dict {}".format(i,roi[i].box_id,roi[i].bufferArrayi,roi[i].__dict__['bufferArrayi']))
        #print("     ")
        
        roi[i].processBuffer()
        
        #print(i,roi[i].box_id,"-",roi[i].bufferArray)
        
        
        if roi[i].checkStopped():
            #print("STOPPED")
            cv2.rectangle(frame,(roi[i].x,roi[i].y),(roi[i].x+roi[i].w,roi[i].y+roi[i].h),(0,0,255),2)
            cv2.putText(frame, 'Stopped', (roi[i].x, roi[i].y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        

        if frameNumber == v.get(cv2.CAP_PROP_FRAME_COUNT):
            print("looping")
            frameNumber = 0 #Or whatever as long as it is the same as next line
            v.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    cv2.imshow('Frame', frame)
    key = cv2.waitKey (5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()