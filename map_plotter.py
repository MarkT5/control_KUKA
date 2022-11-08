import math

import cv2
import numpy as np

log_data = False


class MapPlotter:
    def __init__(self, robot, width=300, map_size=300):
        self.robot = robot
        self.map_size = map_size
        self.bg = np.array([[[190, 70, 20]] * width] * width, dtype=np.uint8)
        self.map_background = np.array([[[100, 100, 100]] * map_size] * map_size, dtype=np.uint8)
        self.map_arr = np.array([[0] * self.map_size] * self.map_size)
        self.width = width
        self.cent_x = map_size // 2
        self.cent_y = map_size // 2
        self.discrete = 20
        if log_data:
            self.log_file = open("lidar_odom_log_8.txt", "a")

    def scale_to_arr(self, x, y):
        return (int(self.map_size / 2 + self.discrete * x), int(self.map_size / 2 - self.discrete * y))

    def create_map(self):
        time = 0
        while cv2.waitKey(1) != 27:
            pos, lidar = self.robot.lidar
            if not pos or not lidar:
                continue
            x, y, ang = pos
            time += 1
            if time < 100:
                continue
            else:
                time = 0

            if log_data:
                self.log_file.write(", ".join(map(str, pos)) + "; " + ", ".join(map(str, lidar)) + "\n")
                continue

            cent_y, cent_x = y, x
            # cv2.circle(self.map_background, scale(1, 0), 4, (255, 255, 255), -1)
            # cv2.circle(self.map_background, scale(0, 1), 4, (255, 0, 255), -1)
            # cv2.circle(self.map_background, scale(cent_x, cent_y), 4, (255, 255, 255), -1)
            cent_y = cent_y - 0.3 * math.cos(ang + math.pi / 2)
            cent_x = cent_x + 0.3 * math.sin(ang + math.pi / 2)
            # cv2.circle(self.map_background, scale(cent_x, cent_y), 4, (0, 255, 0), -1)
            wall_1 = 0
            wall_2 = 0
            wall_len = 30
            current_wall = 1
            corner = None
            for i in range(0, len(lidar)):
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
                # cv2.circle(self.map_background, scale(ox, oy), 4, (0, 0, 0), -1)
                self.map_background[self.scale_to_arr(ox, oy)] = [0, 0, 0]
                self.map_arr[self.scale_to_arr(ox, oy)] = 1
                # detect corners
                if i > 0:
                    if lidar[i - 1] < lidar[i]:
                        if current_wall == 1:
                            wall_1 += 1
                        else:
                            if wall_2 < wall_len:
                                wall_2 = 0
                            current_wall = 1
                            corner = self.scale_to_arr(ox, oy)
                    else:
                        if current_wall == 2:
                            wall_2 += 1
                        else:
                            if wall_1 < wall_len:
                                wall_1 = 0
                            current_wall = 2
                            corner = self.scale_to_arr(ox, oy)

                    if wall_1 >= wall_len and wall_2 >= wall_len:
                        if corner:
                            pass
                            cv2.circle(self.map_background, (corner[1], corner[0]), 2, (0, 0, 255), -1)
                            wall_1, wall_2 = 0, 0
                            # self.map_background[corner] = [0, 0, 255]
                    if lidar[i - 1] - lidar[i] > 1:
                        wall_1, wall_2 = 0, 0
            res = cv2.resize(self.map_background, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("map", res)
        self.log_file.close()

# robot = KUKA('192.168.88.25', ros=False, offline=False)

# new_map = MapPlotter(robot)
# new_map.show_map()
