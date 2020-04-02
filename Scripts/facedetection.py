import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
fce = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
	ret,frame = cap.read()
	frame  = cv2.resize(frame,(600,600))
#fym = cv2.imread('D:\\ncs_demo\\NCS\\Face-Gallery\\ShashankNigam_4.jpg')
	fymb= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow('webcam',fymb)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	face  = fce.detectMultiScale(fymb,scaleFactor=1.3,minNeighbors=6)
	dsp = frame.copy()
	for (x,y,w,h) in face:
		cv2.rectangle(dsp,(x,y),(x+w,y+h),(191,191,200),1)
	#face_d = dsp[y:y+h,x:x+w]
	cv2.imshow('image_marked',dsp)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
cap.release()	
cv2.destroyAllWindows()
