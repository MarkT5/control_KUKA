import time

import cv2
import numpy as np


hu = 255
su = 255
vu = 255
hl = 0
sl = 0
vl = 0
thu = 255
thl = 0
blur = 1


def change_hu(val):
    global hu
    hu = val


def change_su(val):
    global su
    su = val


def change_vu(val):
    global vu
    vu = val


def change_hl(val):
    global hl
    hl = val


def change_sl(val):
    global sl
    sl = val


def change_vl(val):
    global vl
    vl = val


def change_thl(val):
    global thl
    thl = val


def change_blur(val):
    global blur
    blur = max(1, val * 2 + 1)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
vid.set(cv2.CAP_PROP_FPS, 30)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def cerare_trackbar_window():
    blank_image = np.zeros(shape=[3, 1800, 3], dtype=np.uint8)
    cv2.imshow("new",blank_image)
    cv2.createTrackbar('hu', "new", 255, 255, change_hu)
    cv2.createTrackbar('su', "new", 255, 255, change_su)
    cv2.createTrackbar('vu', "new", 255, 255, change_vu)
    cv2.createTrackbar('hl', "new", 0, 255, change_hl)
    cv2.createTrackbar('sl', "new", 0, 255, change_sl)
    cv2.createTrackbar('vl', "new", 0, 255, change_vl)
    cv2.createTrackbar('thl', "new", 0, 255, change_thl)
    cv2.createTrackbar('blur', "new", 0, 10, change_blur)

#lower_bound = np.array([26, 244, 191])
#upper_bound = np.array([21, 153, 120])

#hu = 109
#su = 197
#vu = 138
#hl = 103
#sl = 143
#vl = 90
#thu = 255
#thl = 0
#blur = 1

cerare_trackbar_window()
while 1:
    ret, frame = vid.read()
    #blue_lower_bound = np.array([103, 143, 90])
    #blue_upper_bound = np.array([109, 197, 138])
    # yellow_lower_bound = np.array([14, 17, 172])
    # yellow_upper_bound = np.array([29, 164, 205])
    lower_bound = np.array([hl, sl, vl])
    upper_bound = np.array([hu, su, vu])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    ret, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("camera", res)
    if cv2.waitKey(5) == 27:
        cv2.destroyAllWindows()
        break
