import cv2
import numpy as np
import scipy


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

conv = np.array(False)
#conv = np.array([[8.53725016e-02, -9.49768794e+00, 4.92895853e+03],
#                 [-8.49742837e+00, 3.81768521e+00, 5.58478086e+03],
#                 [-1.30828431e-04, 2.47205973e-03, 1.00000000e+00]])
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


while True:
    ret, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.polylines(frame, [test_rect], True, (255, 255, 255), 1)
    cent_X, cent_Y, back_X, back_Y = find_robot(hsv)
    cv2.circle(frame, (cent_X, cent_Y), 2, (255, 255, 255), -1)
    cv2.circle(frame, (back_X, back_Y), 2, (255, 255, 0), -1)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
