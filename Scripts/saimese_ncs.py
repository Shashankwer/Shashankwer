from __future__ import print_function
import cv2
import sys
import os
from argparse import ArgumentParser,SUPPRESS
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork,IECore

fce=None
database={}
database_path = "D:\\ncs_demo\\NCS\\Siamese_database\\" 
exec_net=None
input_blob = None
output_blob = None

#np.seterr(divide='ignore',invalid='ignore')
def cal_sims(img1_cost,img2_cost):
  #image1_coeff = shared_conv.predict(np.expand_dims(img1,0))
  #image2_coeff = shared_conv.predict(np.expand_dims(img2,0))
  sims = np.inner(img1_cost/np.linalg.norm(img1_cost),img2_cost/np.linalg.norm(img2_cost))
  return sims[0]

def inifce():
	global fce
	fce = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def captureLive():
	inifce()
	global fce
	global exec_net
	global input_blob
	global output_blob
	global database
	cap = cv2.VideoCapture(0)
	while(True):
		start_time = time()
		ret,frame = cap.read()
		fymb= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		face  = fce.detectMultiScale(fymb,scaleFactor=1.3,minNeighbors=6)
		dsp = frame.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		for (x,y,w,h) in face:
			face_d = dsp[y:y+h,x:x+w]
			face_d = cv2.resize(face_d,(100,100))
			face_d = face_d/255
			cv2.imshow('img',face_d)
			face_d = face_d.transpose((2,0,1))
			res = exec_net.infer(inputs = {input_blob:face_d})
			max_dist = -100
			name='unknown'
			for k,v in database.items():
				dist = cal_sims(v,res[output_blob])
				if dist>max_dist:
					max_dist=dist
					name=k
			#print(max_dist,name)		
			if max_dist>0.5:
				name = name.split('.')
				cv2.rectangle(dsp,(x,y),(x+w,y+h),(0,240,0),1)
				cv2.putText(dsp,name[0],(x,y-10),font,1,(0,240,0),1,cv2.LINE_AA)
			else:
				cv2.rectangle(dsp,(x,y),(x+w,y+h),(0,0,240),1)
				cv2.putText(dsp,'unkown',(x,y-10),font,1,(0,0,240),1,cv2.LINE_AA)
		cv2.putText(dsp,f'FPS:{np.round(1.0/(time()-start_time))}',(10,40),font,1,(0,0,240),1,cv2.LINE_AA)		
		#print()		
		cv2.imshow('WebCam',dsp)
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break		
	cap.release()	
	cv2.destroyAllWindows()	
		
def build_argparser():
	parser = ArgumentParser(add_help=False)
	args = parser.add_argument_group('Options')
	args.add_argument('-h','--help',action='help',default=SUPPRESS,help='Show this help message and exit')
	args.add_argument("-m","--model",help="Path to an .xml File with a Trained model.",required = True,type=str)
	args.add_argument("-d","--device",help="Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",default="CPU",type=str)
	args.add_argument("-l","--cpu_extension",help = "Optional. Required for CPU custom layers",default=None,type=str)
	return parser

def create_dict():
	global database
	global database_path
	global exec_net
	global input_blob
	global output_blob
	for i in sorted(os.listdir(database_path)):
		img = cv2.imread(database_path+i)
		img = cv2.resize(img,(100,100))
		img = img/255
		img = img.transpose((2,0,1))
		res = exec_net.infer(inputs = {input_blob:img})
		database[i] =res[output_blob]
	print(database)	
	
	
def main():
	global exec_net
	global input_blob
	global output_blob
	global database
	log.basicConfig(format="[ %(levelname)s ]  %(message)s",level=log.INFO,stream=sys.stdout)
	args = build_argparser().parse_args()
	print(args.model)
	model_xml = args.model
	model_bin = os.path.splitext(model_xml)[0]+'.bin'
	log.info("Creating inference Engine")
	ie = IECore()
	if args.cpu_extension and 'CPU' in args.device:
		ie.add_extension(args.cpu_extension,"CPU")
	log.info("Loading network files:\n \t{}\n\t{}".format(model_xml,model_bin))
	net=IENetwork(model=model_xml,weights=model_bin)
	if "CPU" in args.device:
		supported_layers=ie.query_network(net,"CPU")
		print(f"Supported layers {supported_layers}")
		not_supported_layers=[l for l in net.layers.keys() if l not in supported_layers]
		if len(not_supported_layers)!=0:
			log.error("Few layers are not supported try adding plugins")
			print(not_supported_layers)
			sys.exit(1)
	log.info("Running the inference")
	input_blob = next(iter(net.inputs))
	output_blob = next(iter(net.outputs))	
	exec_net = ie.load_network(network=net,device_name=args.device)
	log.info("Constructing Database")
	create_dict()
	captureLive()
	
if __name__ == '__main__':
		sys.exit(main() or 0)