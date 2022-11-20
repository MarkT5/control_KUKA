import math

import cv2
import numpy as np

log_data = False


class MapPlotter:
    def __init__(self, robot=None, width=300, map_size=170, show_map=True):
        self.robot = robot
        self.show_map = show_map
        self.map_size = map_size
        self.bg = np.array([[[190, 70, 20]] * width] * width, dtype=np.uint8)
        self.map_background = np.array([[[100, 100, 100]] * map_size] * map_size, dtype=np.uint8)
        self.map_arr = np.array([[1] * self.map_size] * self.map_size)
        self.width = width
        self.cent_x = map_size // 2
        self.cent_y = map_size // 2
        self.discrete = 20
        self.pos = (0,0,0)
        if not robot:
            self.from_log()
        if log_data:
            self.log_file = open("lidar_odom_log/lidar_odom_log_9.txt", "a")

    def from_log(self):
        f = open("lidar_odom_log/lidar_odom_log_7.txt", "r")
        txt_log_data = f.read().split('\n')
        log_data = []
        for i in txt_log_data:
            sp_log_data = i.split(';')
            if sp_log_data[-1] == '':
                break
            odom = sp_log_data[0].split(',')
            lidar = sp_log_data[1].split(',')
            odom = list(map(float, odom))
            lidar = list(map(float, lidar))
            log_data.append([odom, lidar])
        self.log_data = log_data[:]
        self.log_data_ind = 0


    def scale_to_arr(self, x, y):
        return (int(self.map_size / 2 + self.discrete * x), int(self.map_size / 2 - self.discrete * y))

    def create_map(self):
        time = 0
        running = True
        while running:
            if self.show_map:
                running = cv2.waitKey(40) != 27
            if self.robot:
                pos, lidar = self.robot.lidar
            else:
                if self.log_data_ind < len(self.log_data)-1:
                    pos, lidar = self.log_data[self.log_data_ind]
                    self.log_data_ind+=1
            if not pos or not lidar:
                continue
            x, y, ang = pos
            self.pos = self.scale_to_arr(*pos[:-1])
            time += 1
            if time < 500 and self.robot:
                continue
            else:
                time = 0

            if log_data:
                self.log_file.write(", ".join(map(str, pos)) + "; " + ", ".join(map(str, lidar)) + "\n")
                continue

            cent_y, cent_x = y, x
            cent_y = cent_y - 0.3 * math.cos(ang + math.pi / 2)
            cent_x = cent_x + 0.3 * math.sin(ang + math.pi / 2)

            for i in range(40, len(lidar)-40):
                lid_ang = i * math.radians(240) / len(lidar) - ang - math.radians(30)
                lid_dist = lidar[i]

                if lid_dist > 5:
                    continue
                for lid_dist_cl in range(int(self.discrete * lid_dist)):
                    ox = cent_x + lid_dist_cl / self.discrete * math.sin(lid_ang)
                    oy = cent_y + lid_dist_cl / self.discrete * math.cos(lid_ang)
                    sx, sy = self.scale_to_arr(ox, oy)
                    if sx > 499 or sy > 499 or sx < 1 or sy < 1:
                        break
                    self.map_background[self.scale_to_arr(ox, oy)] = [255, 255, 255]
                    self.map_arr[self.scale_to_arr(ox, oy)] = 0

                ox = cent_x + lid_dist * math.sin(lid_ang)
                oy = cent_y + lid_dist * math.cos(lid_ang)

                self.map_background[self.scale_to_arr(ox, oy)] = [0, 0, 0]
                self.map_arr[self.scale_to_arr(ox, oy)] = 1
            if self.show_map:
                res = cv2.resize(self.map_background, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("map", res)
        if log_data:
            self.log_file.close()

