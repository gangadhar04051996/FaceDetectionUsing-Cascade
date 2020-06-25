# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:13:34 2020

@author: Gangadhar
"""

import cv2 as cv

# open cv has haar cascades 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
#Defining a function that will do the detection 
#the function works on the images 
#grey is bw image , frame is original image
#Size of the image is reduced to 1.3 times i.e. is the parameter
#in order to accept the particular feature the neighbouring 5 frames should detect it feature as well 
def detect(frame,gray):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
#    Iterate through faces and detect face
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,23,159),1)
#        Two Region of interest one for black and White image to detect eyes
#        Another for the drawing rectange on Colored image
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.5,22)
        smile = smile_cascade.detectMultiScale(roi_gray,1.6,35)
        for (ex,ey,ew,eh) in eyes:
             cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(214,200,120),2)
        for (sx,sy,sw,sh) in smile:
             cv.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(21,20,255),2)
    return frame

#0 is internal webcam and 1 is the external webcam
print("press   q to quit the video frame")
video_capture = cv.VideoCapture(0)
while True:
    _,last_frame =  video_capture.read()
#    BW of frame 
    gray = cv.cvtColor(last_frame,cv.COLOR_BGR2GRAY)
    frame_canvas  = detect(last_frame,gray)
    cv.imshow('Video',frame_canvas)
    if (cv.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release() # Releasing Camera
cv.destroyAllWindows()
