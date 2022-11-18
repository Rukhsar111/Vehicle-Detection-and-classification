import base64
import logging
import os
import random
import shutil
import smtplib
import socket
import ssl
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from threading import Lock, Thread
import math

import cv2
import numpy as np
import torch
from PIL import Image
from pyexpat.errors import XML_ERROR_INCOMPLETE_PE
from torchvision import models








threshold = 0.5


category_index=['car','truck']


class CameraStream(object):
    def __init__(self, src=0):
        # self.stream = cv2.VideoCapture("%s"%RSTP_protocal)
        self.stream = cv2.VideoCapture("Road traffic video for object recognition.mp4") ##########--------

        # fps=self.stream.get(cv2.CAP_PROP_FRAME_COUNT)

        # print("fps",fps)

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()


    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self



    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
            time.sleep(.005)


    def read(self):
        try:
            self.read_lock.acquire()
            frame = self.frame.copy()
            self.read_lock.release()
            return frame
        except:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()

    

    def stop(self):
        self.started = False
        print("entered in stop function")         
        self.thread.join(timeout=1)
        print("exit from stop function")


        


def box_normal_to_pixel(box, dim):
    width, height = dim[0], dim[1]
    box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
    return np.array(box_pixel)




def box_Center(loc):
    """Function finds centroid of detected box.
     
    Args: loc is normalized cordinate of detected object at perticular frame.
    
    return : center of box 
    """
    height=int(loc[3]-loc[1])
    width= int(loc[2]-loc[0])
    center=((loc[0]+(width/2)),(loc[1]+(height/2)))
    return center



def main_function():
    
    track_id=0
    track_object={}
    count=0
    # center_points_cur_frame=[]

    center_points_cur_prev=[]

    while True:
        
        
        time.sleep(.01)
        start_timeing=time.time()

        frame = video_capture.read()

        count+=1

        
        frame=cv2.resize(frame,(620,620))
        
        

        center_points_cur_frame=[]
        # Do the inferencing

        
        pred_model = model(frame)

        print('\n',pred_model)    
        
        boxes = pred_model.xyxy[0][:,:4].cpu()
        scores = pred_model.xyxy[0][:,4].cpu()
        classes = pred_model.xyxy[0][:,5].cpu()
        print(boxes)

        cls = classes.tolist()
        scores = np.array(scores)
        classes = np.array(classes).astype(np.int32)

       

        dets=[]
        veh_class=[]

        if len(classes)!=0:

            for i, value in enumerate(classes):
                # print(value)
                if scores[i] > 0.2: 
                   if int(value)==0  or  int(value)==1:
                        
                        (left, top, right, bottom) = (boxes[i][0],boxes[i][1], boxes[i][2],boxes[i][3])
                        dets.append([left, top, right, bottom])
                        veh_class.append(int(value))

                        p1 = (int(left), int(top))
                        p2 = (int(right), int(bottom))

                        cv2.rectangle(frame, p1, p2, (0,0,255) , 2,1)
                        cv2.putText(frame, category_index[value], ((int(left), int(top)+4)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255) )
                         

                        cv2.imwrite("images"+"/"+ img_name + str(count)+ ".jpg" , frame)
         

        time.sleep(0.1)
        cv2.imshow('img', frame)
        cv2.waitKey(0)


cv2.destroyAllWindows()
#  Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'best_1.pt'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS =  'classes.txt'
# Model Initialization
yolo_file_path = r'C:\Users\Aatif\Desktop\DESKTOP\VIDS\yolov5'
model = torch.hub.load(yolo_file_path, 'custom', path= PATH_TO_FROZEN_GRAPH , source='local',force_reload=True)


names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

print(names)


while True :
    
    video_capture = CameraStream().start()        
    while video_capture.stream.isOpened():
                
        try:            
            
            main_function()

        except Exception as e :
            print("Main function is failing please check error is %s"%e)
            
    video_capture.stop()
    print("video capture has been stoped")
    cv2.destroyAllWindows()
    print("window has been destroyed")
    # traceback.print_exc()
































        


    
    
   
    
    
    


                    




                    

                   
           