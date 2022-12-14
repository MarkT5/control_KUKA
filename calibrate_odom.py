import cv2
import numpy as np


def mouse(event, x, y, flags, param):
    global mouseX, mouseY, vid_coords
    if event == 1:
        mouseX, mouseY = x, y
        if conv.any():
            new = np.dot(conv, np.array([[x, y, 1, 1]]).T)
            print(new / new[3])
        if vid_coords.any():
            vid_coords = np.append(vid_coords, np.array([[x, y, 1, 1]]), axis=0)
        else:
            vid_coords = np.array([[x, y, 1, 1]])


vid_coords = np.array(False)
conv = np.array(False)
corners_coords = np.array([[0, 0, 1, 1], [3.3, 0, 1, 1], [3.3, 3.3, 1, 1], [0, 3.3, 1, 1]])

mouseX, mouseY = 0, 0
vid = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vid.set(cv2.CAP_PROP_FPS, 30)
vid.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
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
print(vid_coords.T)
print(corners_coords.T)
print()
conv = np.dot(corners_coords.T, np.linalg.pinv(vid_coords.T))
conv /= conv[3, 3]
print(conv)
while True:
    ret, frame = vid.read()
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
