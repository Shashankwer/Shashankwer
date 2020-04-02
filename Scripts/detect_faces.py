import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fce = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
i=0
while True:
	ret,frame = cap.read()
	fymb= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow('webcam',fymb)
	face  = fce.detectMultiScale(fymb,scaleFactor=1.3,minNeighbors=6)
	dsp = frame.copy()
	b = False
	for (x,y,w,h) in face:
		cv2.rectangle(dsp,(x,y),(x+w,y+h),(191,191,200),1)
		face_d = dsp[y:y+h,x:x+w]
		face_d = cv2.resize(face_d,(100,100))
		cv2.imshow('image_marked',face_d)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		i=i+1
		cv2.imwrite('D:\\ncs_demo\\NCS\\Siamese_database\\santosh'+str(i+1)+'.jpg',face_d)
		if i==10:
			b=True		
	if b:
		break
		