# ref for the threading increase of FPS:
#   https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# imports
from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import simpleaudio as sa
import math
import argparse
from collections import deque
from timeit import default_timer as timer


""" ***********************************VARIABLE DECLARATION PHASE ***************************************************"""
# construct the argument parse and parse the arguments
# (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=10, help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

# definitions
font = cv2.FONT_HERSHEY_SIMPLEX # font of the text that will appear in the screen
x1 = 0
x2 = 0
y1 = 0
y2 = 0
enddraw = None
wasout = []  # "drum stick was out in the last frame"
region = []
nameregion = []
start1 = []
start2 = []
end1 = []
end2 = []
start3 = []
end3 = []


""" ***********************************FUNCTION DECLARATION PHASE ***************************************************"""
# define function that verifies the distance
def isinsideregion(x_, y_, r):
    if (r[0] <= x_ <= r[2] or r[2] <= x_ <= r[0]) and (r[3] <= y_ <= r[1] or r[1] <= y_ <= r[3]):
        return True
    else:
        return False


# define mouse events
def on_mouse(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        #start1.append(timer())
        #wave_obj[0].play()q
        cv2.putText(frame, "X", x, y)
        #end1.append(timer())

""" ****************************PHASE 2: MIDI SELECTION PHASE *******************************************************"""
# creating audio objects
selectWAVfile = np.array(['prato.wav', 'bumbo.wav', 'bumbo.wav', 'prato.wav'])
wave_obj = []
for i in range(0, len(selectWAVfile)):
    wave_obj.append(sa.WaveObject.from_wave_file(selectWAVfile[i]))  # this will be changed by a MIDI pulse

""" ****************************PHASE 3: DRUM START *****************************************************************"""
""" NO VIDEO DISPLAY IN THIS PHASE - THE GOAL IS TO HAVE THE HIGHER POSSIBLE FPS RATE """
# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.namedWindow('frame')
cv2.imshow('res', frame)
# loop over some frames...this time using the threaded stream
while True:
    start3.append(timer())
    ret, frame = cap.read()

    # set mouse callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse, wave_obj)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('res', frame)
    end3.append(timer())

print("avg time of loop without any sound (ms)")
c3 = []
for i in range(0, len(end2)):
    c3.append(end3[i] - start3[i])
t3 = (sum(c3) / len(end3))
print(t3 * 1000)

print("avg time sound 1 .wav (ms)")
c = []
for i in range(0, len(end1)):
    c.append(end1[i] - start1[i])
t1 = (sum(c)/len(end1))
print(t1*1000)

print("avg time sound 2 .wav (ms)")
c2 = []
for i in range(0, len(end2)):
    c2.append(end2[i] - start2[i])
t2 = (sum(c2)/len(end2))
print(t2*1000)