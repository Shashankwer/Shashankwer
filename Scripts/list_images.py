import os
import cv2

database_path = "D:\\ncs_demo\\NCS\\Siamese_database\\" 

for i in sorted(os.listdir(database_path)):
	print(i)
	print(database_path+i)
	img = cv2.imread(database_path+i)
	cv2.imshow(i,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

