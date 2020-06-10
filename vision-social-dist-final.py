#!/usr/bin/env python3
"""
######## Social Distancing on EdgeTPU - Google Coral Accelerator #########
#
# Author: Rahul Khanna D
# Date: 09-06-2020
# Description: 
# This project uses a TensorFlow Lite model to perform people detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# You can find the entire documentation of this project at:
# https://github.com/
# Credits: Adrian, Evan Juras
"""

import os
import time
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from scipy.spatial import distance as dist

MIN_DISTANCE = 50 #min safe dist (in pixels) that two people can be from each other. 
NEAR_DISTANCE = 75 #dist (in pixels) that two people are nearby. 

"""
    Parse command line arguments.
    
    :return: command line arguments
"""
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--modeldir", required=True, type=str,
                        help="Folder path to the .tflite file.")

parser.add_argument("-g", "--graph", type=str,
                    help="Name of the .tflite file, if different than detect.tflite.",
                    default='detect.tflite')

parser.add_argument("-l", "--labels", type=str,
                    help="Name of the labelmap file, if different than labelmap.txt.",
                    default='labelmap.txt')

parser.add_argument("-i", "--input", required=True, type=str,
                    help="Path to image or video file or CAM.")

parser.add_argument("-pt", "--threshold", help='Probability threshold for detection filtering',
                    default=0.5)

parser.add_argument('--edgetpu',
                    help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
INPUT_NAME = args.input
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()
#print(CWD_PATH)

im_flag = False

if(INPUT_NAME == 'CAM'):
    INPUT_PATH = 0
    
elif(INPUT_NAME.endswith('.jpg') or INPUT_NAME.endswith('.bmp')):
    #INPUT_PATH = os.path.join(CWD_PATH,INPUT_NAME)
    INPUT_PATH = CWD_PATH + INPUT_NAME
    im_flag = True
    
else:# Path to video file
    INPUT_PATH = os.path.join(CWD_PATH,INPUT_NAME)
    INPUT_PATH = CWD_PATH + INPUT_NAME
    im_flag = False
    
print(INPUT_PATH)
base = os.path.basename(INPUT_NAME)
filename = os.path.splitext(base)[0]
#print(filename)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    #print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

if im_flag == True:
    print('img')
    frame = cv2.imread(INPUT_PATH)
    imH, imW, _ = frame.shape 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    start = time.perf_counter()
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    inference_time = time.perf_counter() - start
    
    p_boxes = []
    p_centroids = []
    p_scores = []
    res = []
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0))and labels[int(classes[i])] == "person":
        #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0))and labels[int(classes[i])] == "person":
            print(classes[i])
            print("Label:", end=" ")
            print(labels[int(classes[i])], end=", ")
            print("Confidence:", end=" ")
            print(scores[i])
            
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            print(xmax - xmin)
            print(ymax - ymin)
            area = (xmax - xmin) * (ymax - ymin)
            print("Area:", end=" ")
            print(area)
            
            centerX = (xmin+xmax)/2
            centerY = (ymin+ymax)/2
            
            p_boxes.append([xmin, ymin, xmax, ymax])
            p_centroids.append((centerX, centerY))
            p_scores.append(float(scores[i]))
            r = (p_scores, p_boxes, p_centroids)
            res.append(r)
    violate = set()
    nearby = set()
    #print(res)
    if len(res) >= 2:
        cent = np.array([r[2] for r in res])
        print(cent[0])
        D = dist.cdist(cent[0], cent[1], metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i, j] < MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)
                elif D[i, j] < NEAR_DISTANCE:
                    nearby.add(i)
                    nearby.add(j)
                    
    for (i, (prob, bbox, centroid)) in enumerate(res):
        print(bbox)
        print(prob)
        print(centroid)
        
        xmin, ymin, xmax, ymax = bbox[i]
        (cX, cY) = centroid[i]
        color = (0, 255, 0)
        
        if i in violate:
            color = (0, 0, 255)
        
        if i in nearby:
            color = (0, 165, 255)                  
        
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 4)
        cv2.circle(frame, (int(cX), int(cY)), 5, color, 1)
        safe = len(res)-len(nearby)-len(violate)
        if safe < 0:
            safe = 0
            
        # Draw label
        text = "Total People detected: {}".format(len(res))
        text1 = "People in Safe distance: {}".format(safe)
        text2 = "People in Alert zone : {}".format(len(nearby))
        text3 = "People Very Close: {}".format(len(violate))
        text4 = "The Lonely Programmer"
        text5 = "Inference Time: {:.2f}ms".format(inference_time*1000)
        print('%.2f ms' % (inference_time * 1000))
        
        cv2.rectangle(frame, (5, frame.shape[0] - 150), (380, frame.shape[0]), (0, 0, 0), cv2.FILLED) # Draw Black box to put label text in
        cv2.putText(frame, text, (10, frame.shape[0] - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
        cv2.putText(frame, text1, (10, frame.shape[0] - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 3)
        cv2.putText(frame, text2, (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 3)
        cv2.putText(frame, text3, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
        cv2.rectangle(frame, (frame.shape[1]-275, frame.shape[0] - 25), (frame.shape[1], frame.shape[0]), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, text4, (frame.shape[1]-270, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        cv2.rectangle(frame, (0, 0), (330, 30), (255,255, 255), cv2.FILLED) # Draw Black box to put label text in
        cv2.putText(frame, text5, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    print('-------------------------------------')
    # All the results have been drawn on the frame, so it's time to display it.
    #frame = cv2.resize(frame, (int(imW/4), int(imH/4)), interpolation = cv2.INTER_AREA)
    writefile = "out-" + filename + ".jpg"
    cv2.imshow('Social Distancing', frame)
    cv2.imwrite(writefile,frame)
    cv2.destroyAllWindows()
    
else:
    print('video')
    # Open video file
    print(INPUT_PATH)
    video = cv2.VideoCapture(INPUT_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_video = int(video.get(cv2.CAP_PROP_FPS))
    print(imH, imW)
    writefile = "out-" + filename + ".mp4"
    out_video = cv2.VideoWriter(writefile, cv2.VideoWriter_fourcc(*'avc1'),fps_video,(int(imW),int(imH)),True)
    while(video.isOpened()):

        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = video.read()
        if not ret:
          print('Output video is generated in the directory!')
          break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        start = time.perf_counter()
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        
        inference_time = time.perf_counter() - start
        
        p_boxes = []
        p_centroids = []
        p_scores = []
        res = []
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0))and labels[int(classes[i])] == "person":
            #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0))and labels[int(classes[i])] == "person":
                print(classes[i])
                print("Label:", end=" ")
                print(labels[int(classes[i])], end=", ")
                print("Confidence:", end=" ")
                print(scores[i])
                
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                print(xmax - xmin)
                print(ymax - ymin)
                area = (xmax - xmin) * (ymax - ymin)
                print("Area:", end=" ")
                print(area)
                
                if(INPUT_PATH != 'CAM' and area > 90000):
                    break
                
                centerX = (xmin+xmax)/2
                centerY = (ymin+ymax)/2
                
                p_boxes.append([xmin, ymin, xmax, ymax])
                p_centroids.append((centerX, centerY))
                p_scores.append(float(scores[i]))
                r = (p_scores, p_boxes, p_centroids)
                res.append(r)
                
        violate = set()
        nearby = set()
        #print(res)
        if len(res) >= 2:
            cent = np.array([r[2] for r in res])
            print(cent[0])
            D = dist.cdist(cent[0], cent[1], metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i+1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
                    elif D[i, j] < NEAR_DISTANCE:
                        nearby.add(i)
                        nearby.add(j)
                        
        for (i, (prob, bbox, centroid)) in enumerate(res):
            print(bbox)
            print(prob)
            print(centroid)
            
            xmin, ymin, xmax, ymax = bbox[i]
            (cX, cY) = centroid[i]
            color = (0, 255, 0)
            
            if i in violate:
                color = (0, 0, 255)
            
            if i in nearby:
                color = (0, 165, 255)                  
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 4)
            cv2.circle(frame, (int(cX), int(cY)), 5, color, 1)
            safe = len(res)-len(nearby)-len(violate)
            if safe < 0:
                safe = 0
                
            # Draw label
            text = "Total People detected: {}".format(len(res))
            text1 = "People in Safe distance: {}".format(safe)
            text2 = "People in Alert zone : {}".format(len(nearby))
            text3 = "People Very Close: {}".format(len(violate))
            text4 = "The Lonely Programmer"
            text5 = "Inference Time: {:.2f}ms".format(inference_time*1000)
            print('%.2f ms' % (inference_time * 1000))
            
            cv2.rectangle(frame, (5, frame.shape[0] - 150), (380, frame.shape[0]), (0, 0, 0), cv2.FILLED) # Draw Black box to put label text in
            cv2.putText(frame, text, (10, frame.shape[0] - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
            cv2.putText(frame, text1, (10, frame.shape[0] - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 3)
            cv2.putText(frame, text2, (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 3)
            cv2.putText(frame, text3, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            
            cv2.rectangle(frame, (frame.shape[1]-275, frame.shape[0] - 25), (frame.shape[1], frame.shape[0]), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, text4, (frame.shape[1]-270, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            cv2.rectangle(frame, (0, 0), (330, 30), (255,255, 255), cv2.FILLED) # Draw Black box to put label text in
            cv2.putText(frame, text5, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        print('-------------------------------------')
        # All the results have been drawn on the frame, so it's time to display it.
        #frame = cv2.resize(frame, (int(imW/4), int(imH/4)), interpolation = cv2.INTER_AREA)
        cv2.imshow('Social Distancing', frame)
        out_video.write(frame)
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()