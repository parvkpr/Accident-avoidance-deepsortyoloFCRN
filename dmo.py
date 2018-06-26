#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
from timeit import time
import tensorflow as tf
from matplotlib import pyplot as plt
import models
import subprocess
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def create1():
    model_data_path="/home/kheteshr/Desktop/deep_sort_yolov3-master/NYU_ResNet-UpProj.npy"
    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    '''\img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
    '''
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
  
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        #saver = tf.train.Saver()     
        #saver.restore(sess, model_data_path)

        # Use to load from npy file
        net.load(model_data_path, sess) 

        # Evalute the network for the given image
        
        print('in func1')
        # Plot result
        '''fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()'''
        sess.close()
        return net,input_node
def predict2(net,input_node,image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
       
       
    with tf.Session() as sess:
        # Evalute the network for the given image
        sess.run(tf.global_variables_initializer())
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        sess.close()
        return pred

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None                  
    nms_max_overlap = 1.0



   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    his = [0]*100001
    writeVideo_flag = True 
    ne,input_node = create1()
    video_capture =  cv2.VideoCapture('s.mp4')
    count=0
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        t1 = time.time()
	
        image = Image.fromarray(frame)
	cv2.imwrite("/home/kheteshr/Desktop/deep_sort_yolov3-master/in/frame%d.jpg" % count, frame)
        pred = predict2(ne,input_node,'in/frame%d.jpg'%count)
        count+=1
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        fps = 1 
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        #print(os.getcwd())
        
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue             
            bbox = track.to_tlbr()
            xi=bbox[0]+bbox[2]
            xi/=20
            yi=bbox[1]+bbox[3]
	    yi*=160
	    yi/=1440		
            a = int(xi)
            b = int(yi) 
            f = (his[track.track_id]-pred[0,a,b,0])/fps
            his[track.track_id]=pred[0,a,b,0]
            print(repr(track.track_id),repr(pred[0,a,b,0])) 
            print('speed'+repr(track.track_id),repr(f)) 
            #predict.predict('NYU_FCRN.ckpt','/home/kheteshr/Desktop/deep_sort_yolov3-master/in/frame%d.jpg'%(count-1),xi,yi)
            # maina(xi,yi,'/home/kheteshr/Desktop/deep_sort_yolov3-master/in/frame%d.jpg'%(count-1))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        als=0    
        cv2.imshow('', frame)
        cv2.imwrite("/home/kheteshr/Desktop/deep_sort_yolov3-master/ot/frameo%d.jpg" % als, frame)
        als+=1
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
