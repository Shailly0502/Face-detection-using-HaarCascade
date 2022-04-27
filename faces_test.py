import numpy as np
import cv2 as cv
import os

haar_cascade=cv.CascadeClassifier('haar_face.xml')

face_recogniser=cv.face.LBPHFaceRecognizer_create()
face_recogniser.read('face_trained.yml')

people=[]
for i in os.listdir(r'D:\opencv\train'):
     people.append(i)
print(people) 

img=cv.imread('IMG-20211206-WA0014.jpg')
img=cv.resize(img,(300,500))
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('testing image', gray)

faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
for(x,y,w,h) in faces_rect:
    face=gray[y:y+h,x:x+w]
    label,confidence=face_recogniser.predict(face)
    print(people[label])
    print(confidence)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
cv.imshow('detected face', img)
cv.waitKey(0)    

