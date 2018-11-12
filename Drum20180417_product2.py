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
from playsound import PlaySound

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
            nameregion.append('nomeinst')
            wasout.append(True)
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
selectWAVfile = np.array(['prato.wav', 'bumbo.wav', 'bumbo.wav', 'prato.wav'])
wave_obj = []
for i in range(0, len(selectWAVfile)):
    wave_obj.append(sa.WaveObject.from_wave_file(selectWAVfile[i]))  # this will be changed by a MIDI pulse

""" ****************************PHASE 3: DRUM START *****************************************************************"""
""" NO VIDEO DISPLAY IN THIS PHASE - THE GOAL IS TO HAVE THE HIGHER POSSIBLE FPS RATE """
# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
cap = VideoStream(src=0).start()
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while True:

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 200 pixels - the smaller the screen, the higher the FPS
    frame = cap.read()
    #frame = imutils.resize(frame, width=200)

    # color tracking of the stick
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # BLUE COLOR
    lowerColor = np.array([110, 50, 50])  # lower boundary of the hsv color
    upperColor = np.array([130, 255, 255])  # upper boundary of the hsv color

    # RED COLOR
    #lowerColor = np.array([0,100,100])  # lower boundary of the hsv color
    #upperColor = np.array([20, 255, 255])  # upper boundary of the hsv color
    mask = cv2.inRange(hsv, lowerColor, upperColor)

    # erosion and dilation for noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # (x, y) center of the ball (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x1, y1), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    # (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # if either of the two last points are None, ignore them
        if pts[0] is None or pts[1] is None:
            continue

        # velocity tracking
        d1 = pts[0]
        d2 = pts[1]
        v = math.sqrt(math.pow((d2[0] - d1[0]), 2) + math.pow((d2[1] - d1[1]), 2))

    # Checking region contact
    for i in range(0, len(region)):
        if isinsideregion(int(x1), int(y1), region[i]) and wasout[i]:
            wasout[i] = False
            play_obj = wave_obj[i].play() # make a thread for that
            #PlaySound(selectWAVfile[i], intensity=3).start()# sometimes is crashing
        elif not isinsideregion(int(x1), int(y1), region[i]):
            wasout[i] = True

    # Show frame
    #cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
cap.stop()