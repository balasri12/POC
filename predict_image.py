import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description=
        'Use this script to run Mask-RCNN object detection and semantic '
        'segmentation network from TensorFlow Object Detection API.')
# parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# parser.add_argument('--model', required=True, help='Path to a .pb file with weights.')
# parser.add_argument('--config', required=True, help='Path to a .pxtxt file contains network configuration.')
parser.add_argument('--classes', help='Optional path to a text file with names of classes.')
parser.add_argument('--colors', help='Optional path to a text file with colors for an every class. '
                                     'An every color is represented with three values from 0 to 255 in BGR channels order.')
parser.add_argument('--width', type=int, default=800,
                    help='Preprocess input image by resizing to a specific width.')
parser.add_argument('--height', type=int, default=800,
                    help='Preprocess input image by resizing to a specific height.')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
args = parser.parse_args()

np.random.seed(324)

input = r'C:\Users\Balajisri\Documents\mobiles\images\template.jpg'
model = r'D:\Documents\Segmentation\paper-data\object_detection_api\inference_graph2\frozen_inference_graph.pb'
config = r'D:\Documents\Segmentation\paper-data\object_detection_api\inference_graph2\forzen_config.pbtxt'

# Load a network
net = cv.dnn.readNetFromTensorflow(model,config)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

frame = cv.imread(input)

frameH = frame.shape[0]
frameW = frame.shape[1]

# Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, size=(args.width, args.height), swapRB=True, crop=False)

# Run a model
net.setInput(blob)

boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

numClasses = masks.shape[1]
numDetections = boxes.shape[2]

for i in range(numDetections):

    box = boxes[0, 0, i]
    mask = masks[i]
    score = box[2]
    print(score)
    if score > args.thr:
        print('here')
        classId = int(box[1])
        left = int(frameW * box[3])
        top = int(frameH * box[4])
        right = int(frameW * box[5])
        bottom = int(frameH * box[6])

        crop_img = frame[top:bottom, left:right]
        cv.imwrite('out.jpg', crop_img)