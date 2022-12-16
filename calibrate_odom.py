import cv2
import numpy as np
import scipy
import math
from KUKA import KUKA
import time

robot = KUKA('192.168.88.23', ros=False, offline=False)


def mouse(event, x, y, flags, param):
    global mouseX, mouseY, vid_coords
    if event == 1:
        mouseX, mouseY = x, y
        if conv.any():
            new = np.dot(conv, np.array([[x, y, 1]]).T)
            print(new / new[-1])
        if vid_coords.any():
            vid_coords = np.append(vid_coords, np.array([[x, y, 1]]), axis=0)
        else:
            vid_coords = np.array([[x, y, 1]])


vid_coords = np.array(False)

correction = [0, 0, 0]

corners_coords = np.array([[0, 0, 1], [3300, 0, 1], [3300, 3300, 1], [0, 3300, 1]])
mouseX, mouseY = 0, 0
vid = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
vid.set(cv2.CAP_PROP_FPS, 30)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

blue_lower_bound = np.array([71, 70, 70])
blue_upper_bound = np.array([133, 246, 200])
yellow_lower_bound = np.array([12, 66, 87])
yellow_upper_bound = np.array([29, 255, 255])

# conv = np.array(False)
conv = np.array([[6.28302929e-02, -6.28928085e+00, 3.40162119e+03],
                 [-6.11168532e+00, 2.51195914e+00, 4.81927110e+03],
                 [3.16546358e-05, 1.56344310e-03, 1.00000000e+00]])
if not conv.any():
    while True:
        ret, frame = vid.read()
        if vid_coords.any():
            for i in range(vid_coords.shape[0]):
                cv2.circle(frame, (vid_coords[i, 0], vid_coords[i, 1]), 5, (255, 0, 0), -1)
            if vid_coords.shape[0] == 4:
                break
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # calculate HOMOGRAPHY
    A = []
    for i in range(4):
        xs, ys, _ = vid_coords[i, :]
        xd, yd, _ = corners_coords[i, :]
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys, -xd])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys, -yd])

    A = np.array(A)
    conv = scipy.linalg.eig(A.T @ A)[1][:, -1]
    conv /= conv[-1]
    conv = conv.reshape(3, 3)
    print(conv)

# draw test rectangle
test_rect = (np.linalg.inv(conv) @ corners_coords.T).T
tr_del = test_rect[:, 2]
test_rect = test_rect[:, :2]
tr1, tr2, tr3, tr4 = test_rect
test_rect = np.array([tr1 / tr_del[0], tr2 / tr_del[1], tr3 / tr_del[2], tr4 / tr_del[3]], np.int32)


def find_robot(hsv):
    mask = cv2.inRange(hsv, blue_lower_bound, blue_upper_bound)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    ret, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
    M = cv2.moments(mask)
    try:
        cent_X = int(M["m10"] / M["m00"])
        cent_Y = int(M["m01"] / M["m00"])
    except:
        cent_X = 0
        cent_Y = 0
    mask = cv2.inRange(hsv, yellow_lower_bound, yellow_upper_bound)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    ret, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
    M = cv2.moments(mask)
    try:
        back_X = int(M["m10"] / M["m00"])
        back_Y = int(M["m01"] / M["m00"])
    except:
        back_X = 0
        back_Y = 0
    return cent_X, cent_Y, back_X, back_Y


def find_robot_abs():
    cent_X, cent_Y, back_X, back_Y = find_robot(hsv)
    r_coord = np.array([cent_X, cent_Y, 1, back_X, back_Y, 1]).reshape((2, 3))
    # print(r_coord.T)
    r_coord = np.dot(conv, (r_coord.T)).T
    yfc, xfc, den_f, yrc, xrc, den_r = r_coord.reshape(6)
    xfc /= den_f
    yfc /= den_f
    xrc /= den_r
    yrc /= den_r
    ang = math.atan2(yfc - yrc, xfc - xrc)
    return xfc, yfc, ang, xrc, yrc


centering = 1
cent_cycles = 5

inv_rob_mat = np.array(False)
def center_robot():
    global centering
    global inv_rob_mat

    if not robot.going_to_target_pos:
        print("check pos")
        time.sleep(1)
        x, y, ang, _, _ = find_robot_abs()
        cr = math.cos(ang)
        sr = math.sin(ang)
        x -= 1650
        y -= 1650
        if centering == 1:
            inv_rob_mat = np.linalg.inv(np.array([[cr, -sr, x / 1000],
                                                  [sr, cr, y / 1000],
                                                  [0, 0, 1]]))
            aco = math.acos(max(-1, min(1, inv_rob_mat[0, 0])))
            asi = math.asin(max(-1, min(1, inv_rob_mat[1, 0])))
            asi_sign = -1 + 2 * (asi > 0)
            xi, yi, angi = np.array([*inv_rob_mat[:2, 2], aco * (asi_sign)]).astype(float)
            print("correct error")
            print(xi, yi, angi)
            robot.go_to(xi, yi, angi)
        elif centering > 1:
            err_mat = np.linalg.inv(np.array([[cr, -sr, x / 1000],
                                                  [sr, cr, y / 1000],
                                                  [0, 0, 1]]))
            err_mat = inv_rob_mat @ err_mat
            aco = math.acos(max(-1, min(1, err_mat[0, 0])))
            asi = math.asin(max(-1, min(1, err_mat[1, 0])))
            asi_sign = -1 + 2 * (asi > 0)
            xi, yi, angi = np.array([*err_mat[:2, 2], aco * (asi_sign)]).astype(float)
            print("correct post error")
            print(xi, yi, angi)
            robot.go_to(xi, yi, angi)
        centering += 1


def rand_pos():
    x, y, ang = np.random.random(3)
    x = (x - 0.5) * 3
    y = (y - 0.5) * 3
    ang = (ang - 0.5) * 2 * 3.1415
    return x, y, ang


x, y, z = rand_pos()
while True:
    xr, yr, zr, = robot.increment
    ret, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.polylines(frame, [test_rect], True, (255, 255, 255), 1)
    cent_X, cent_Y, back_X, back_Y = find_robot(hsv)
    xfc, yfc, ang, xrc, yrc = find_robot_abs()

    if not 3300 > xfc > 0 or not 3300 > yfc > 0 and not centering:
        robot.go_to(0, 0, 0)
        print("centering")
        centering = 1
    if cent_cycles > centering > 0:
        center_robot()
    elif centering == cent_cycles:
        if not robot.going_to_target_pos:
            del robot
            time.sleep(3)
            robot = KUKA('192.168.88.23', ros=True, offline=False)
            time.sleep(2)
            centering = 0

    if not robot.going_to_target_pos and not centering:
        centering = False
        print("inc:", xr, yr, zr)
        print("camera:", x, y, z)
        print("error", x - xr, y - yr, z - zr)
        x, y, z = rand_pos()
        print(x, y, z)
        robot.go_to(x, y, z)

    cv2.putText(frame, f"x: {round(xfc - 1650)}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"y: {round(yfc - 1650)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"x2: {round(xr * 1000)}", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"y2: {round(yr * 1000)}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"ang2: {zr}", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ang: {round(ang, 5)}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cent_X, cent_Y), 2, (255, 255, 255), -1)
    cv2.circle(frame, (back_X, back_Y), 2, (255, 255, 0), -1)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
