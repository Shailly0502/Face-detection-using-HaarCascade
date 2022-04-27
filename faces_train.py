import os
import cv2 as cv
import numpy as np

people=[]
for i in os.listdir(r'D:\opencv\train'):
     people.append(i)
print(people) 
dir=r"D:\opencv\train"
features=[]
labels=[]
haar_cascade=cv.CascadeClassifier('haar_face.xml')
def create_train():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for(x,y,w,h) in faces_rect:
                faces=gray[y:y+h, x:x+w]
                features.append(faces)
                labels.append(label)

create_train()

features=np.array(features)
labels=np.array(labels)

face_recogniser=cv.face.LBPHFaceRecognizer_create()
face_recogniser.train(features,labels)
np.save('features.npy', features,allow_pickle=True)
np.save('labels.npy', labels,allow_pickle=True)
face_recogniser.save('face_trained.yml')
