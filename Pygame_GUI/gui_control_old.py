import math

import cv2
import numpy as np
from pynput import keyboard

deb = True


def debug(inf):
    if deb:
        print(inf)


class GuiControl:
    def __init__(self, width, height, robot):

        # robot class
        self.robot = robot

        # window properties
        self.window_name = 'arm_sim'
        self.width = width
        self.height = height

        # window settings
        self.update_sliders = True
        self.to_draw_reachable = True
        self.control_mode = 1
        self.cylindrical_scale = 1.5
        self.start_point_x = int(width // 2 - 30 * self.cylindrical_scale)
        self.start_point_y = int(height // 2 - 60 * self.cylindrical_scale)
        self.move_body_scale = 100

        # canvases
        self.arm_background = np.array([[[190, 70, 20]] * width] * height, dtype=np.uint8)
        self.arm_screen = np.copy(self.arm_background)

        self.body_pos_background = np.array([[[190, 70, 20]] * 1000] * 1000, dtype=np.uint8)
        self.body_pos_screen = np.copy(self.body_pos_background)

        # arm parameters
        self.m2_ang_offset = - math.pi
        self.m3_ang_offset = 2 * math.pi
        self.m4_ang_offset = 0
        self.m2_len = 155
        self.m3_len = 135
        self.m4_len = 200
        self.m2_range = [-65, 90]
        self.m3_range = [-150, 146]

        # operable data
        self.target = [[100, 100], math.pi / 2]
        self.target_cartesian = [[50, 50, 50], math.pi / 2]
        self.move_speed = [0.0, 0.0, 0.0]
        self.move_speed_val = 0.3
        self.arm_height = 50
        self.pressed_keys = []
        self.mouse_coords = [0, 0]
        self.target_body_pos = [0, 0, 0]

        # flags, counters, service
        self.update_target = False
        self.arm_update_counter = 0
        self.update_move_speed = False
        self.send_command_counter = 0
        self.speed_updated = False
        self.m1_ang_speed = 0
        self.old_lidar = None
        self.old_body_pos = [0, 0, 0]

    def init_direct_control_sliders(self):
        cv2.createTrackbar('m2', self.window_name, int(self.robot.arm_pos[1]+180), 360, self.change_m2_angle)
        cv2.createTrackbar('m3', self.window_name, int(self.robot.arm_pos[2]+180), 360, self.change_m3_angle)
        cv2.createTrackbar('m4', self.window_name, int(self.robot.arm_pos[3]+180), 360, self.change_m4_angle)

    def init_cartesian_control_sliders(self):
        cv2.createTrackbar('height', self.window_name, 50, 100, self.change_height)

    def start_listener(self):
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()

    def can_update_arm(self):
        self.arm_update_counter += 1
        if self.arm_update_counter > 0:
            self.arm_update_counter = 0
            return True
        else:
            return False

    def renew(self):
        cv2.destroyWindow(self.window_name)
        self.run()

    def run(self):
        cv2.imshow(self.window_name, self.arm_screen)
        if self.control_mode == 0:
            self.init_direct_control_sliders()
            self.control_mode = 3

        cv2.imshow("pos", self.body_pos_screen)
        cv2.imshow(self.window_name, self.arm_screen)

        cv2.setMouseCallback(self.window_name, self.change_pos)
        cv2.setMouseCallback("pos", self.go_to_pos)
        self.start_listener()

        while 1:
            cv2.imshow("camera", self.robot.cam_image_BGR)

            self.arm_screen = np.copy(self.arm_background)
            self.update_arm(self.cylindrical_scale)
            self.update_body_pos()
            self.update_lidar()
            self.update_keys()
            if self.control_mode == 1:
                self.draw_target()

            cv2.imshow("pos", self.body_pos_screen)
            cv2.imshow(self.window_name, self.arm_screen)

            k = cv2.waitKey(10)

            # exit program
            if k == 27:
                cv2.destroyAllWindows()
                self.robot.disconnect()
                break

            # change control mode
            elif k == 49 and self.control_mode != 1:
                debug("Control mode changed to cylindrical")
                self.control_mode = 1
                self.renew()
            elif k == 50 and self.control_mode != 2:
                debug("Control mode changed to cartesian")
                self.control_mode = 2
                self.renew()
            elif k == 51 and self.control_mode != 3:
                debug("Control mode changed to direct")
                self.control_mode = 0
                self.renew()

            # grabber
            elif k == 103:
                self.robot.arm_pos[-1] = 0
                self.robot.grab(self.robot.arm_pos[-1])
            elif k == 114:
                self.robot.arm_pos[-1] = 2
                self.robot.grab(self.robot.arm_pos[-1])

            # speed control

    def change_m2_angle(self, val):
        if self.can_update_arm:
            self.robot.move_arm(m2=val - 180)

    def change_m3_angle(self, val):
        if self.can_update_arm:
            self.robot.move_arm(m3=val - 180)

    def change_m4_angle(self, val):
        if self.can_update_arm:
            self.robot.move_arm(m4=val - 180)

    def change_height(self, val):
        if self.can_update_arm:
            self.robot.move_arm(m5=val - 180)

    def update_keys(self):
        fov = 0
        if 'w' in self.pressed_keys:
            fov += 1
        if 's' in self.pressed_keys:
            fov -= 1
        self.move_speed[0] = fov * self.move_speed_val

        rot = 0
        if 'a' in self.pressed_keys:
            rot += 1
        if 'd' in self.pressed_keys:
            rot -= 1
        self.move_speed[2] = rot * self.move_speed_val

        side = 0
        if 'q' in self.pressed_keys:
            side += 1
        if 'e' in self.pressed_keys:
            side -= 1
        self.move_speed[1] = side * self.move_speed_val

        arm_rot = 0
        if 'z' in self.pressed_keys:
            arm_rot += 0.5
        if 'x' in self.pressed_keys:
            arm_rot -= 0.5

        if self.update_move_speed:
            self.robot.move_base(*self.move_speed)
            self.robot.going_to_target_pos = False
            self.update_move_speed = False

        if arm_rot != 0 and self.can_update_arm():
            self.robot.move_arm(m1=max(min(self.robot.arm_pos[0] + arm_rot, 134), -157))

    def on_press(self, key):
        try:
            if key.char not in self.pressed_keys:
                self.pressed_keys.append(key.char)
                self.update_move_speed = True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key.char in self.pressed_keys:
                self.pressed_keys.pop(self.pressed_keys.index(key.char))
                self.update_move_speed = True
        except AttributeError:
            pass

    def update_lidar(self):
        buff, lidar = self.robot.lidar

        if buff and len(buff) == 3:
            x, y, ang = buff
            if lidar:
                if self.old_lidar == lidar:
                    x, y, ang = self.old_body_pos
                else:
                    self.old_body_pos = buff
                    self.old_lidar = lidar
                cent_y, cent_x = y * self.move_body_scale + 500, -x * self.move_body_scale + 500
                cent_y = int(cent_y - 30 * math.cos(ang + math.pi / 2))
                cent_x = int(cent_x - 30 * math.sin(ang + math.pi / 2))
                for l in range(len(lidar)):
                    cv2.ellipse(self.body_pos_screen,
                                (cent_y, cent_x),
                                (int(lidar[l] * self.move_body_scale), int(lidar[l] * self.move_body_scale)),
                                math.degrees(ang), int(-180 / 170 * l), int(-180 / 170 * (l + 1)), (0, 0, 255), 5)

    def update_body_pos(self):
        self.body_pos_screen = np.copy(self.body_pos_background)
        buff = self.robot.increment
        if buff:
            x, y, ang = self.target_body_pos
            cv2.circle(self.body_pos_screen, (x, y), 3, (100, 255, 100), -1)
            x, y, ang = self.robot.increment
            cv2.circle(self.body_pos_screen,
                       (int(y * self.move_body_scale + 150), int(-x * self.move_body_scale + 150)), 5, (255, 255, 255),
                       -1)
            size = 30 * self.move_body_scale // 100
            xl1 = int(size * math.cos(ang + math.pi / 2))
            yl1 = int(size * math.sin(ang + math.pi / 2))
            xl2 = int(size * math.cos(ang + math.pi / 2))
            yl2 = int(size * math.sin(ang + math.pi / 2))
            size = 20 * self.move_body_scale // 100
            xw1 = int(size * math.cos(ang))
            yw1 = int(size * math.sin(ang))
            xw2 = int(size * math.cos(ang))
            yw2 = int(size * math.sin(ang))

            x1 = int(y * self.move_body_scale + xl1 + xw1 + 150)
            y1 = int(-x * self.move_body_scale + yl1 + yw1 + 150)
            x2 = int(y * self.move_body_scale - xl2 + xw2 + 150)
            y2 = int(-x * self.move_body_scale - yl2 + yw2 + 150)
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
            x1 = int(y * self.move_body_scale + xl1 - xw1 + 150)
            y1 = int(-x * self.move_body_scale + yl1 - yw1 + 150)
            x2 = int(y * self.move_body_scale - xl2 - xw2 + 150)
            y2 = int(-x * self.move_body_scale - yl2 - yw2 + 150)
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)

            x1 = int(y * self.move_body_scale + xw1 + xl1 + 150)
            y1 = int(-x * self.move_body_scale + yw1 + yl1 + 150)
            x2 = int(y * self.move_body_scale - xw2 + xl2 + 150)
            y2 = int(-x * self.move_body_scale - yw2 + yl2 + 150)
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
            x1 = int(y * self.move_body_scale + xw1 - xl1 + 150)
            y1 = int(-x * self.move_body_scale + yw1 - yl1 + 150)
            x2 = int(y * self.move_body_scale - xw2 - xl2 + 150)
            y2 = int(-x * self.move_body_scale - yw2 - yl2 + 150)
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)

    def update_arm(self, scale=1.0):

        m1_ang, m2_ang, m3_ang, m4_ang, m5_ang, grab = *map(math.radians, self.robot.arm_pos[:-1]), self.robot.arm_pos[-1]
        color = (255, 100, 100)

        for i in range(2):
            m2_ang += self.m2_ang_offset
            m3_ang += self.m3_ang_offset
            m4_ang += self.m4_ang_offset

            m3_ang += m2_ang
            m4_ang += m3_ang
            m2x = self.start_point_x
            m2y = self.height - self.start_point_y
            m3x = int(m2x + self.m2_len * math.sin(m2_ang) / scale)
            m3y = int(m2y + self.m2_len * math.cos(m2_ang) / scale)
            m4x = int(m3x + self.m3_len * math.sin(m3_ang) / scale)
            m4y = int(m3y + self.m3_len * math.cos(m3_ang) / scale)
            m5x = int(m4x + self.m4_len * math.sin(m4_ang) / scale)
            m5y = int(m4y + self.m4_len * math.cos(m4_ang) / scale)

            cv2.line(self.arm_screen, (m2x, m2y), (m3x, m3y), color, 2)
            cv2.line(self.arm_screen, (m3x, m3y), (m4x, m4y), color, 2)
            cv2.line(self.arm_screen, (m4x, m4y), (m5x, m5y), color, 2)
            try:
                if self.robot.arm[0]:
                    m1_ang, m2_ang, m3_ang, m4_ang, m5_ang = map(math.radians, self.robot.arm)
                else:
                    break
                color = (255, 255, 255)
            except:
                break


    def draw_target(self):
        target = [[(self.mouse_coords[0] - self.start_point_x) * self.cylindrical_scale,
                   (-self.mouse_coords[1] + self.height - self.start_point_y) * self.cylindrical_scale], self.target[1]]
        _, _, _, available = self.robot.solve_arm(target)
        if self.update_target:
            color = ((0, 230, 0) if available else (0, 0, 230))
        else:
            color = ((100, 255, 100) if available else (100, 100, 240))
        cv2.circle(self.arm_screen, (int(self.mouse_coords[0]), int(self.mouse_coords[1])), 10, color, 4)
        size = 20
        xs = int(self.mouse_coords[0] + size * 0.6 * math.cos(-self.target[1] - math.pi / 2))
        ys = int(self.mouse_coords[1] + size * 0.6 * math.sin(-self.target[1] - math.pi / 2))
        xe = int(self.mouse_coords[0] - size * 1.4 * math.cos(-self.target[1] - math.pi / 2))
        ye = int(self.mouse_coords[1] - size * 1.4 * math.sin(-self.target[1] - math.pi / 2))
        cv2.line(self.arm_screen, (xs, ys), (xe, ye), color, 2)

    def draw_reachable(self):
        m2_range = [-63, 84]
        m3_range = [-110, 135]
        m4_range = [-120, 60]
        center = [0, 0]
        center[0] = int(self.start_point_x + self.m4_len * math.sin(self.target[1]) / self.cylindrical_scale)
        center[1] = int(
            self.height - self.start_point_y + self.m4_len * math.cos(self.target[1]) / self.cylindrical_scale)
        m2_len = int(self.m2_len / self.cylindrical_scale)
        m3_len = int(self.m3_len / self.cylindrical_scale)
        color = (0, 0, 255)
        cv2.ellipse(self.arm_screen, center, (m2_len + m3_len, m2_len + m3_len), 0, m2_range[0] - 90, m2_range[1] - 90,
                    color,
                    -1)
        cent_l = list(map(int, (
            center[0] + m2_len * math.sin(math.radians(m2_range[0])),
            center[1] - m2_len * math.cos(math.radians(m2_range[0])))))
        cv2.ellipse(self.arm_screen, cent_l, (m3_len, m3_len), 0, m3_range[0] + m2_range[0] - 90,
                    m3_range[1] + m2_range[0] - 90,
                    color, -1)
        cent_l = list(map(int, (
            center[0] + m2_len * math.sin(math.radians(m2_range[1])),
            center[1] - m2_len * math.cos(math.radians(m2_range[1])))))
        cv2.ellipse(self.arm_screen, cent_l, (m3_len, m3_len), 0, m3_range[0] + m2_range[1] - 90,
                    m3_range[1] + m2_range[1] - 90,
                    color, -1)

    def go_to_pos(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_body_pos = [int(x), int(y), 0]
            x, y = (x - 150) / self.move_body_scale, (-y + 150) / self.move_body_scale
            self.robot.go_to(y, x)

    def change_pos(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.update_target = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.update_target = True

        if self.control_mode == 1:
            self.inverse_kinematics_cylindrical_control(event, x, y, flags)
        elif self.control_mode == 2:
            self.inverse_kinematics_cartesian_control(event, x, y, flags)

    def inverse_kinematics_cylindrical_control(self, event, x, y, flags):
        self.mouse_coords = [x, y]
        if event == cv2.EVENT_MOUSEMOVE or self.update_target:
            if self.update_target:
                self.target[0] = [(x - self.start_point_x) * self.cylindrical_scale,
                                  (-y + self.height - self.start_point_y) * self.cylindrical_scale]
                if self.can_update_arm:
                    self.robot.move_arm(target=self.target)

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.target[1] -= 0.1
            else:
                self.target[1] += 0.1
            if self.target[1] > 2 * math.pi:
                self.target[1] -= 2 * math.pi
            elif self.target[1] < 0:
                self.target[1] += 2 * math.pi

    def inverse_kinematics_cartesian_control(self, event, x, y, flags):
        if event == cv2.EVENT_MOUSEMOVE:
            if self.update_target:
                cv2.circle(self.arm_screen, (x, y), 10, (255, 255, 0), -1)
                self.target_cartesian[0][:2] = [x, y]
                if self.can_update_arm:
                    self.robot.move_arm(target=self.target)

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.target_cartesian[0][2] += 10
            else:
                self.target_cartesian[0][2] -= 10
