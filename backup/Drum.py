# imports
import cv2
import numpy as np
import simpleaudio as sa
import math
import imutils
import argparse
from collections import deque


# define function for distance calculation
# def calculatedistance(x1, y1, x2, y2):
    # dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # return dist

def f():
    print('oi')


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


# construct the argument parse and parse the arguments
# (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

# beginning of the i
cap = cv2.VideoCapture(0)

# definitions
font = cv2.FONT_HERSHEY_SIMPLEX # font of the text that will appear in the screen
x1 = 0
x2 = 0
y1 = 0
y2 = 0
enddraw = None
regionx1 = []
regionx2 = []
regiony1 = []
regiony2 = []
nameregion = []

# PHASE 1: Beginning of the identification phase
while True:
    ret, frame = cap.read()

    # User orientation message
    cv2.putText(frame, 'Selecione a area do instrumento (quando terminar pressione t)', (20, 50),
                font, 1, (255, 0, 0), 2, cv2.LINE_4)  # write the drum instrument text

    # set mouse callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse, frame)

    # draw in the image
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # verifies if the draw ended
    if enddraw == 1:
        cv2.putText(frame, 'Confirma a regiao? s para confirmar', (x2, y2),
                    font, 1, (255, 0, 0), 2, cv2.LINE_4)  # write the drum instrument text
        if cv2.waitKey(1) & 0xFF == ord('s'):
            regionx1.append(int(x1))
            regiony1.append(int(y1))
            regionx2.append(int(x2))
            regiony2.append(int(y2))
            nameregion.append('nomeinst')
            cv2.putText(frame, 'salvando região...', (x2+100, y2),
                        font, 1, (0, 0, 255), 2, cv2.LINE_4)  # write the drum instrument text


    # draw the already chosen rectangles in the screen
    for i in range(0, len(regionx1)):
        # cv2.circle(frame, (selectlocationx[i],selectlocationy[i]), selectlocationradius[i], (0, 0, 0), 5)
        cv2.rectangle(frame, (regionx1[i], regiony1[i]), (regionx2[i], regiony2[i]), (0, 0, 0), 5)
        cv2.putText(frame, nameregion[i], (regionx1[i], regiony2[i]+20), font, 2,
                    (0, 255, 0), 2, cv2.LINE_4)  # write the drum instrument text

    # show image
    cv2.imshow('frame', frame)

    # Criar uma GUI para a seleção das regiões e acompanhamento dos instrumentos

    # exit mode
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# selecting a region of pixels to be an instrument
# selectlocationx = np.array([200, 400, 700, 900]) # ***this will be replaced by a selection on screen***
# selectlocationy = np.array([200, 400, 400, 200]) # ***this will be replaced by a selection on screen***
# selectlocationradius = np.array([80, 80, 100, 90]) # ***this will be replaced by a selection on screen***
# selectname = np.array(['prato', 'caixa', 'bumbo', 'prato']) # ***this will be replaced by a selection on screen**


# PHASE 2: MIDI audio selection

# popup a window to choose the midi output for each region generated

# creating audio objects
selectWAVfile = np.array(['prato.wav', 'bumbo.wav', 'bumbo.wav', 'prato.wav'])
wave_obj = []
for i in range(0, len(selectWAVfile)):
    wave_obj.append(sa.WaveObject.from_wave_file(selectWAVfile[i])) # this will be changed by a MIDI pulse

# plays the audio: just for testing
# play_obj = wave_obj[0].play()
# play_obj = wave_obj[1].play()
# play_obj.wait_done()

# PHASE 3: Drum start
while True:
    ret, frame = cap.read()

    # color tracking of the stick
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerColor = np.array([110, 50, 50])  # lower boundary of the hsv color
    upperColor = np.array([130, 255, 255])  # upper boundary of the hsv color
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

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x1), int(y1)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    # (https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # if either of the two last points are None, ignore them
        if pts[0] is None or pts[1] is None:
            continue

        # velocity tracking
        d1 = pts[0]
        d2 = pts[1]
        v = math.sqrt(math.pow((d2[0] - d1[0]), 2) + math.pow((d2[1] - d1[1]), 2))

        cv2.putText(frame, str(v), (150, 150), font, 1, (255, 0, 0), 2, cv2.LINE_4)  # write the drum instrument text

        if v < 1:
            cv2.putText(frame, 'Bateu', (600, 150),
                        font, 1, (255, 0, 0), 2, cv2.LINE_4)  # write the drum instrument text

    # Checking region contact and draw images
    for i in range(0, len(regionx1)):
        if regionx1[i] <= x1 <= regiony1[i] and regionx2[i] <= y1 <= regiony2[i]:
            play_obj = wave_obj[i].play()

        # if calculatedistance(posX, posY, regionx1[i], regiony1[i]) < selectlocationradius[i]:
        # play_obj = wave_obj[i].play()

        # Draw regions
        # cv2.circle(frame, (selectlocationx[i],selectlocationy[i]), selectlocationradius[i], (0, 0, 0), 5)
        cv2.rectangle(frame, (regionx1[i], regiony1[i]), (regionx2[i], regiony2[i]), (0, 0, 0), 5)
        cv2.putText(frame, nameregion[i], (regionx1[i],regiony2[i]+20), font, 2,
                    (0, 255, 0), 2, cv2.LINE_4)  # write the drum instrument text

    cv2.imshow('res', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
