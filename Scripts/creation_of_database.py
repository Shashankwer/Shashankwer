import cv2
import numpy as np
import matplotlib.pyplot as plt

#cap = cv2.VideoCapture(0)
fce = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#while(True):
#ret,frame = cap.read()
#frame  = cv2.resize(frame,(600,600))
fym = cv2.imread('D:\\ncs_demo\\NCS\\Siamese_database\\Shashank.jpg')
fymb= cv2.cvtColor(fym,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',fymb)
cv2.waitKey(0)
cv2.destroyAllWindows()
face  = fce.detectMultiScale(fymb,scaleFactor=1.3,minNeighbors=6)
dsp = fym.copy()
for (x,y,w,h) in face:
	cv2.rectangle(dsp,(x,y),(x+w,y+h),(191,191,200),1)
	face_d = dsp[y:y+h,x:x+w]
	face_d = cv2.resize(face_d,(100,100))
	cv2.imshow('image_marked',face_d)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('D:\\ncs_demo\\NCS\\Siamese_database\\Shashank.jpg',face_d)
#cap.release()	
#cv2.destroyAllWindows()
