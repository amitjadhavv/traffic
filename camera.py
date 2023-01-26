import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from PIL import Image
import tensorflow as tf
#import tflite_runtime.interpreter as tflite
def classify(image):
    # image = Image.open(file_path)
    # image = image.resize((40,40))
    # print(image.shape)
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    #print(image.shape)
    pred = model.predict([image])
    #sign = classes[pred]
    return pred

classes={0: 'Speed limit 20 kilometer per hour',
 1: 'Speed limit 50 kilometer per hour',
 2: 'Speed limit 70 kilometer per hour',
 3: 'Speed limit 80 kilometer per hour',
 4: 'No passing',
 5: 'Right-of-way at intersection',
 6: 'Priority road',
 7: 'Give Way',
 8: 'Stop',
 9: 'No vehicles',
 10: 'Vehicles over 3.5 tons prohibited',
 11: 'No entry',
 12: 'General caution',
 13: 'Dangerous curve left',
 14: 'Dangerous curve right',
 15: 'Double curve',
 16: 'Bumpy road',
 17: 'Road narrows on the right',
 18: 'Road work',
 19: 'Children crossing',
 20: 'Wild animals crossing',
 21: 'Turn right ahead',
 22: 'Turn left ahead',
 23: 'Ahead only',
 24: 'Go straight or right',
 25: 'Go straight or left',
 26: 'Keep right',
 27: 'Keep left',
 28: 'Roundabout mandatory'}

class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        (self.grabbed, self.frame) = self.stream.read()


        self.stopped = False

    def start(self):

        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:

            if self.stopped:

                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):

        return self.frame

    def stop(self):
        self.stopped = True

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',default='640x480')
args = parser.parse_args()
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
print(imW,imH)
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
model=tf.keras.models.load_model("model2.h5")
# interpreter = tflite.Interpreter('model1.tflite')
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#height = input_details[0]['shape'][1]
#width = input_details[0]['shape'][2]
#floating_model = (input_details[0]['dtype'] == np.float32)
frame_rate_calc = 1
freq = cv2.getTickFrequency()

while True:
    t1 = cv2.getTickCount()
    frame = videostream.read()
    try:
        frame1 = frame.copy()
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    #frame_resized = cv2.resize(frame_rgb, (height,width))
        img = cv2.resize(frame_rgb, (640,480))
        windows = slide_window(img, x_start_stop=[0, 500], y_start_stop=[0,],xy_window=(40,40))
        # print(len(windows))
        for window in windows:
            img_cropped =  img[window[0][1]:window[1][1],window[0][0]:window[1][0],:]
            # print(img_cropped.shape)# Extract single window
            pred=classify(img_cropped)
            if max(pred[0])>0.99:
                print(classes[np.argmax(pred[0])])
                time.sleep(2)
                break
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        #print(frame_rate_calc)
    except:
        pass
#     input_data = np.expand_dims(frame_resized, axis=0)
#     input_data = (np.float32(input_data) - input_mean) / input_std
#     interpreter.set_tensor(input_details[0]['index'],input_data)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     data=np.argmax(output_data[0])
#     print(classes[data],max(output_data[0]))
#     if data>=0.9:
#     	print(classes[np.argmax(output_data[0])+1])
# cv2.destroyAllWindows()
videostream.stop()
