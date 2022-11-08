import time

import cv2
import numpy as np

from KUKA import KUKA

robot = KUKA(ros=False)

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


def cerare_trackbar_window():
    cv2.imshow("new", robot.cam)
    cv2.createTrackbar('hu', "new", 255, 255, change_hu)
    cv2.createTrackbar('su', "new", 255, 255, change_su)
    cv2.createTrackbar('vu', "new", 255, 255, change_vu)
    cv2.createTrackbar('hl', "new", 0, 255, change_hl)
    cv2.createTrackbar('sl', "new", 0, 255, change_sl)
    cv2.createTrackbar('vl', "new", 0, 255, change_vl)
    cv2.createTrackbar('thl', "new", 0, 255, change_thl)
    cv2.createTrackbar('blur', "new", 0, 10, change_blur)


m1_ang = 0
m4_ang = -90
update = 0
flag_x = 0
flag_y = 0
cube_x = 0
cube_y = 0


k = 0.001
try:
    time.sleep(1)
    robot.move_arm(0, 20, -80, -90, 0, 2)
    time.sleep(3)
    while 1:
        height, width = robot.cam.shape[:2]
        # lower_bound = np.array([hl, sl, vl])
        # upper_bound = np.array([hu, su, vu])

        lower_bound = np.array([11, 125, 84])
        upper_bound = np.array([23, 255, 255])

        hsv = cv2.cvtColor(robot.cam, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        ret, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
        try:
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(robot.cam, (cX, cY), 5, (255, 255, 255), -1)

            cube_x = cX - width / 2
            cube_y = -cY + height / 2
        except:
            cube_x = 0
            cube_y = 0

        speed_x = cube_x*k
        speed_y = cube_y*k
        m1_ang = min(m1_ang - speed_x, 180)
        m4_ang = max(m4_ang + speed_y, -90)
        robot.move_arm(m1=m1_ang, m4=m4_ang)
        update = 0
        cv2.imshow("camera", robot.cam)
        update += 1
        if cv2.waitKey(5) == 27:
            cv2.destroyAllWindows()
            robot.disconnect()
            break

except Exception as e:
    print(e)
    cv2.destroyAllWindows()
    time.sleep(2)
    robot.disconnect()
