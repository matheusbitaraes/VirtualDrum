# ref for the threading increase of FPS:
#   https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# imports
from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import pyaudio
import numpy as np
import math
import struct
from HearBeat import HearBeat

""" ***********************************VARIABLE DECLARATION PHASE ***************************************************"""
# construct the argument parse and parse the arguments
# (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
'''ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=10, help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])'''

# definitions
font = cv2.FONT_HERSHEY_SIMPLEX  # font of the text that will appear in the screen
x1 = 0
x2 = 0
x_2 = 0
y1 = 0
y2 = 0
y_2 = 0
enddraw = None
wasout_1 = []  # "Blue drum stick was out in the last frame"
wasout_2 = []  # "Red drum stick was out in the last frame"
region = []
nameregion = []

# audio analysis variables
#CHUNK = 2048
#CHUNK = 1024
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

fs = 44100       # sampling rate, Hz, must be integer
duration = 0.5   # in seconds, may be float
f = 750.0        # first instrument sine frequency, Hz, may be float


""" ***********************************FUNCTION DECLARATION PHASE ***************************************************"""


# define function that verifies the distance
def isinsideregion(x_, y_, r):
    if (r[0] <= x_ <= r[2] or r[2] <= x_ <= r[0]) and (r[3] <= y_ <= r[1] or r[1] <= y_ <= r[3]):
        return True
    else:
        return False


# define mouse events
def on_mouse(event, x, y, flags, frame):
    global x1, x2, y1, y2, enddraw

    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        x2 = x
        y2 = y
        enddraw = 0

    elif event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        enddraw = 1

    elif event == cv2.EVENT_MOUSEMOVE and enddraw == 0:
        x2 = x
        y2 = y


""" ****************************PHASE 1: REGION IDENTIFICATION PHASE ************************************************"""
cap = cv2.VideoCapture(0)
num_inst = 1
while True:
    ret, frame = cap.read()

    # User orientation message
    cv2.putText(frame, 'Selecione a area do instrumento (quando terminar pressione e segure t)', (20, 50),
                font, 1, (255, 0, 0), 2, cv2.LINE_4)  # write the drum instrument text

    # set mouse callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse, frame)

    # draw in the image
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # verifies if the draw ended
    if enddraw == 1:
        cv2.putText(frame, 'Presione e segure s para confirmar', (x2, y2),
                    font, 1, (255, 0, 0), 2, cv2.LINE_4)  # write the drum instrument text
        if cv2.waitKey(1) & 0xFF == ord('s'):
            region.append([int(x1), int(y1), int(x2), int(y2)])
            nameregion.append('Inst.' + str(num_inst))
            num_inst = num_inst + 1
            wasout_1.append(True)
            wasout_2.append(True)
            enddraw = 2

    # draw the already chosen rectangles in the screen
    for i in range(0, len(region)):
        # cv2.circle(frame, (selectlocationx[i],selectlocationy[i]), selectlocationradius[i], (0, 0, 0), 5)
        cv2.rectangle(frame, (region[i][0], region[i][1]), (region[i][2], region[i][3]), (0, 0, 0), 5)
        cv2.putText(frame, nameregion[i], (region[i][0], (region[i][3])+20), font, 2,
                    (0, 255, 0), 2, cv2.LINE_4)  # write the drum instrument text
    # show image
    cv2.imshow('frame', frame)

    # Criar uma GUI para a seleção das regiões e acompanhamento dos instrumentos

    # exit mode
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break
cv2.destroyAllWindows()

""" ****************************PHASE 2: MIDI SELECTION PHASE *******************************************************"""
# creating audio objects
'''hb1 = HearBeat(CHUNK, FORMAT, CHANNELS, RATE, fs)
hb1.pause()
hb1.start()
hb2 = HearBeat(CHUNK, FORMAT, CHANNELS, RATE, fs)
hb2.pause()
hb2.start()'''

# generate samples for each instrument, note conversion to float32 array
samples = []
hb1 = []
hb2 = []
for k in range(num_inst):
    hb1.append(HearBeat(CHUNK, FORMAT, CHANNELS, RATE, fs))
    hb2.append(HearBeat(CHUNK, FORMAT, CHANNELS, RATE, fs))
    samples.append((np.sin(2*np.pi*np.arange(fs*duration)*(f+k*60)/fs)).astype(np.float32))
    hb1[k].start()
    hb1[k].setSamples(samples[k])
    hb2[k].start()
    hb2[k].setSamples(samples[k])

""" ****************************PHASE 3: DRUM START *****************************************************************"""

""" NO VIDEO DISPLAY IN THIS PHASE - THE GOAL IS TO HAVE THE HIGHER POSSIBLE FPS RATE """
# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
cap = VideoStream(src=0).start()
fps = FPS().start()
x1 = 0
y1 = 0
# loop over some frames...this time using the threaded stream
while True:

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 200 pixels - the smaller the screen, the higher the FPS
    frame = cap.read()

    # color tracking of the stick
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # BLUE COLOR
    lowerColor_1 = np.array([110, 50, 50])  # lower boundary of the hsv color
    upperColor_1 = np.array([130, 255, 255])  # upper boundary of the hsv color

    # RED COLOR
    lowerColor_2 = np.array([0, 130, 100])  # lower boundary of the hsv color
    upperColor_2 = np.array([20, 255, 255])  # upper boundary of the hsv color

    # masks creation
    mask_1 = cv2.inRange(hsv, lowerColor_1, upperColor_1)
    mask_2 = cv2.inRange(hsv, lowerColor_2, upperColor_2)
    cv2.imshow("mask", mask_1)

    # erosion and dilation for noise removal
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask_2 = cv2.morphologyEx(mask_2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    cv2.imshow("transformed mask", mask_1)

    # (x, y) center of the ball (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
    contours_1 = cv2.findContours(mask_1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours_2 = cv2.findContours(mask_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #center = None

    # only proceed if at least one contour was found
    if len(contours_1) > 0:
        # find the largest contour in the mask_1, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours_1, key=cv2.contourArea)
        ((x1, y1), radius) = cv2.minEnclosingCircle(c)
        #M = cv2.moments(c)
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if len(contours_2) > 0:
        c = max(contours_2, key=cv2.contourArea)
        ((x_2, y_2), radius) = cv2.minEnclosingCircle(c)

    # update the points queue
    #pts.appendleft(center)

    '''# Checking sound
    data = stream.read(CHUNK, exception_on_overflow=False)
    energy = rms(data)
    if energy > 0.009: # energy threshold for drum activation
        # Checking region contact
        for i in range(0, len(region)):

            # Blue stick
            #if isinsideregion(int(x1), int(y1), region[i]) and wasout_1[i]:
            if isinsideregion(int(x1), int(y1), region[i]):
                wasout_1[i] = False
                playsound(i, energy*10)
            elif not isinsideregion(int(x1), int(y1), region[i]):
                wasout_1[i] = True

            # Red stick
            #if isinsideregion(int(x_2), int(y_2), region[i]) and wasout_2[i]:
            if isinsideregion(int(x_2), int(y_2), region[i]):
                wasout_2[i] = False
                playsound(i, energy*10)
            elif not isinsideregion(int(x_2), int(y_2), region[i]):
                wasout_2[i] = True'''

    # Checking region contact
    for i in range(0, len(region)):

        # Blue stick
        if isinsideregion(int(x1), int(y1), region[i]) and wasout_1[i]:
            wasout_1[i] = False
            # start a thread for audio listening
            #hb1.setSamples(samples[i])
            hb1[i].retake()
        elif not isinsideregion(int(x1), int(y1), region[i]):
            wasout_1[i] = True
            # kill thread fo audio listening
            if not hb1[i].paused:
                hb1[i].pause()

        # Red stick
        if isinsideregion(int(x_2), int(y_2), region[i]) and wasout_2[i]:
            wasout_2[i] = False
            #hb2.setSamples(samples[i])
            hb2[i].retake()
        elif not isinsideregion(int(x_2), int(y_2), region[i]):
            wasout_2[i] = True
            # kill thread
            if not hb2[i].paused:
                hb2[i].pause()

    # Show frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
for k in range(num_inst):
    hb1[k].stop()
    hb2[k].stop()
    hb1[k].terminate()
    hb2[k].terminate()
cv2.destroyAllWindows()
cap.stop()


