import math
import time

import cv2
import numpy as np

from Pygame_GUI.Objects import *
from Pygame_GUI.Screen import Screen
from KUKA import KUKA

deb = True


def debug(inf):
    if deb:
        print(inf)


class GuiControl:
    def __init__(self, width, height, robot):

        # robot class
        if isinstance(robot, list):
            self.robots_num = len(robot)
            self.robots = robot
            self.robot = KUKA(robot[0])
            self.curr_robot = 0
        else:
            self.robot = robot
            self.robots = []
            self.robots_num = 0

        # window properties
        self.width = width
        self.height = height

        # arm window settings
        self.cylindrical_scale = (1 / self.height)*1000
        self.arm_width = int(0.4839*self.width)
        self.arm_height = int(self.height*0.6154)
        self.start_point_x = int(self.arm_width // 2 - 30 / self.cylindrical_scale)
        self.start_point_y = int(self.arm_height // 2 - 50 / self.cylindrical_scale)
        self.move_body_scale = 30
        self.economy_mode = False

        # canvases
        self.arm_background = np.array([[[20, 70, 190]] * int(0.4839*self.width)] * int(self.height*0.6154), dtype=np.uint8)
        self.arm_screen = np.copy(self.arm_background)

        self.body_pos_background = np.array([[[20, 70, 190]] * int(0.2419*self.width)] * int(self.height*0.3846), dtype=np.uint8)
        self.body_pos_screen_wh = [int(0.2419*self.width)//2, int(self.height*0.3846)//2]
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
        self.move_speed_val = 0.5
        self.target_body_pos = [0, 0, 0]

        # flags, counters, service
        self.old_lidar = None
        self.old_body_pos = [0, 0, 0]
        self.last_checked_pressed_keys = None
        self.robot.going_to_pos_sent = False
        self.current_cam_mode = True
        self.grip_pos = 0.01

    def init_pygame(self):
        """
        Initialises PyGame and precreated pygame objects:
        two buttons to change camera mode and six sliders to control arm
        """
        if self.robot.connected:

            while not self.robot.arm:
                time.sleep(0.05)
            m1_ang, m2_ang, m3_ang, m4_ang, m5_ang, grip = *self.robot.arm, 0
        else:
            m1_ang, m2_ang, m3_ang, m4_ang, m5_ang, grip = 0, 0, 0, 0, 0, 0
        self.screen = Screen(self.width, self.height)
        Button(self.screen, x=0.6, y=0.897, width=0.08, height=0.06, color=(150, 255, 170), func=self.change_cam_mode)
        Button(self.screen, x=0.726, y=0.897, width=0.08, height=0.06, color=(150, 255, 170), func=self.print_arm)
        self.m1_slider = Slider(self.screen,
                                min=157, max=-134, val=m1_ang,
                                x=0.556, y=0.641,
                                width=0.403, height=0.025,
                                color=(150, 160, 170),
                                func=self.change_m1_angle)
        self.m2_slider = Slider(self.screen,
                                min=-84, max=63, val=m2_ang,
                                x=0.556, y=0.679,
                                width=0.403, height=0.025,
                                color=(150, 160, 170),
                                func=self.change_m2_angle)
        self.m3_slider = Slider(self.screen, min=-135, max=110, val=m3_ang,
                                x=0.556, y=0.718,
                                width=0.403, height=0.025,
                                color=(150, 160, 170),
                                func=self.change_m3_angle)
        self.m4_slider = Slider(self.screen, min=-90, max=95, val=m4_ang,
                                x=0.556, y=0.756,
                                width=0.403, height=0.025,
                                color=(150, 160, 170),
                                func=self.change_m4_angle)
        self.m5_slider = Slider(self.screen, min=-145, max=96, val=m5_ang,
                                x=0.556, y=0.795,
                                width=0.403, height=0.025,
                                color=(150, 160, 170),
                                func=self.change_m5_angle)
        self.grip_slider = Slider(self.screen, min=0.001, max=1.98, val=grip,
                                  x=0.556, y=0.833,
                                  width=0.403, height=0.025,
                                  color=(150, 160, 170),
                                  func=self.change_grip)

        self.robot_cam_pygame = Mat(self.screen, x=0, y=0, width=0.5261, height=0.6154, cv_mat_stream=self.cam_stream)
        self.body_pos_pygame = Mat(self.screen, x=0, y=0.615, cv_mat_stream=self.body_pos_stream,
                                   func=self.update_body_pos)
        self.arm_pygame = Mat(self.screen, x=0.516, y=0, cv_mat_stream=self.arm_stream, func=self.mouse_on_arm)
        self.pos_text_x = Text(self.screen,
                               x=0.024, y=0.949,
                               inp_text=self.output_pos_text_x,
                               font='serif',
                               font_size=10)
        self.pos_text_y = Text(self.screen,
                               x=0.105, y=0.949,
                               inp_text=self.output_pos_text_y,
                               font='serif',
                               font_size=10)

    def print_arm(self, *args):
        print(self.robot.arm_pos)

    def change_cam_mode(self, *args):
        """
        When called changes camera mode to different from current
        """
        self.current_cam_mode = not self.current_cam_mode

    def cam_stream(self, *args):
        """
        service function for correct work with CvMat
        :return: map CvMat
        """

        if self.current_cam_mode:
            return self.robot.camera_BGR()
        else:
            return self.robot.depth_camera()

    def change_robot(self, *args):
        self.curr_robot += 1
        del self.robot
        if self.curr_robot >= self.robots_num:
            self.curr_robot = 0
        if self.robots_num > 1:
            self.robot = KUKA(self.robots[self.curr_robot])
            self.current_cam_mode = not self.current_cam_mode
            self.change_cam_mode()

    def body_pos_stream(self):
        """
        service function for correct work with CvMat
        :return: map CvMat
        """
        return self.body_pos_screen

    def arm_stream(self):
        """
        service function for correct work with CvMat
        :return: manipulator control CvMat
        """
        return self.arm_screen

    def run(self):
        """
        main cycle
        initialises PyGame, updates all data, check pressed keys, updates screen
        :return:
        """
        self.init_pygame()
        while self.screen.running:
            pass
            if not self.economy_mode:
                self.update_arm(self.cylindrical_scale)
            self.update_keys()
            self.screen.step()
        self.robot.disconnect()

    def change_m1_angle(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: angle of joint 1
        :return:
        """
        self.robot.move_arm(m1=val)

    def change_m2_angle(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: angle of joint 2
        :return:
        """
        self.robot.move_arm(m2=val)

    def change_m3_angle(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: angle of joint 3
        :return:
        """
        self.robot.move_arm(m3=val)

    def change_m4_angle(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: angle of joint 4
        :return:
        """
        self.robot.move_arm(m4=val)

    def change_m5_angle(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: angle of joint 5
        :return:
        """
        self.robot.move_arm(m5=val)

    def change_grip(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: grip position
        :return:
        """
        self.robot.move_arm(grip=val)

    def output_pos_text_x(self):
        odom = self.robot.increment_data
        if odom:
            return "x:{}".format(round(odom[0], 2))
        else:
            return "No Data"

    def output_pos_text_y(self):
        odom = self.robot.increment_data
        if odom:
            return "y:{}".format(round(odom[1], 2))
        else:
            return ""

    # def change_height(self, val):
    #    self.robot.move_arm(m5=val)

    def update_keys(self):
        """
        checks pressed keys and configure commands to send according to pressed keys
        :return:
        """
        arm_rot = self.robot.arm[0]
        grip = self.grip_pos
        pressed_keys = self.screen.pressed_keys
        fov = 0
        if pg.K_w in pressed_keys:
            fov += 1
        if pg.K_s in pressed_keys:
            fov -= 1
        self.move_speed[0] = fov * self.move_speed_val

        rot = 0
        if pg.K_a in pressed_keys:
            rot += 1
        if pg.K_d in pressed_keys:
            rot -= 1
        self.move_speed[2] = rot * self.move_speed_val

        side = 0
        if pg.K_q in pressed_keys:
            side += 1
        if pg.K_e in pressed_keys:
            side -= 1

        if pg.K_x in pressed_keys:
            arm_rot -= 1
        if pg.K_z in pressed_keys:
            arm_rot += 1

        if pg.K_g in pressed_keys:
            if grip < 1:
                grip = 1.98
            else:
                grip = 0.01


        self.move_speed[1] = side * self.move_speed_val
        if self.last_checked_pressed_keys != pressed_keys:
            self.robot.move_base(*self.move_speed)
            self.robot.going_to_target_pos = False
            self.last_checked_pressed_keys = pressed_keys[:]
            if self.grip_pos != grip:
                self.robot.move_arm(grip=grip)
                self.grip_pos = grip
        if self.robot.arm[0] != arm_rot:
            self.robot.move_arm(m1=arm_rot)


    def update_lidar(self):
        """
        draws lidar data on body_pos_screen
        :return:
        """
        buff, lidar = self.robot.lidar
        if buff and len(buff) == 3:
            x, y, ang = buff
            if lidar:
                if self.old_lidar == lidar:
                    x, y, ang = self.old_body_pos
                else:
                    self.old_body_pos = buff
                    self.old_lidar = lidar
                cent_y, cent_x = y * self.move_body_scale + self.body_pos_screen_wh[0], -x * self.move_body_scale + self.body_pos_screen_wh[0]
                cent_y = int(cent_y - 0.3 * self.move_body_scale * math.cos(ang + math.pi / 2))
                cent_x = int(cent_x - 0.3 * self.move_body_scale * math.sin(ang + math.pi / 2))
                for l in range(0, len(lidar), 5):
                    if not 0.01 < lidar[l] < 5.5:
                        continue
                    color = (0, max(255, 255 - int(45.5 * l)), min(255, int(45.5 * l)))
                    cv2.ellipse(self.body_pos_screen, (cent_y, cent_x),
                                (int(lidar[l] * self.move_body_scale), int(lidar[l] * self.move_body_scale)),
                                math.degrees(ang), 30 + int(-240 / len(lidar) * l),
                                30 + int(-240 / len(lidar) * (l + 1)), color,
                                max(1, int(0.1 * self.move_body_scale)))

    def update_body_pos(self, *args):
        """
        draws body rectangle on body_pos_screen and sends robot to set position if mouse pressed
        :param args: set: relative mouse position and is mouse pressed
        :return:
        """
        if self.economy_mode:
            return
        pos = args[0]
        state = args[1]
        if state:
            if not self.robot.going_to_pos_sent:
                self.go_to_pos(*pos)
                self.robot.going_to_pos_sent = True
        else:
            self.robot.going_to_pos_sent = False
        self.body_pos_screen = np.copy(self.body_pos_background)
        buff = self.robot.increment
        if buff:
            x, y, ang = self.target_body_pos
            cv2.circle(self.body_pos_screen, (x, y), 3, (100, 255, 100), -1)
            x, y, ang = self.robot.increment
            cv2.circle(self.body_pos_screen,
                       (int(y * self.move_body_scale + self.body_pos_screen_wh[0]), int(-x * self.move_body_scale + self.body_pos_screen_wh[1])),
                       max(1, int(0.05 * self.move_body_scale)), (255, 255, 255), -1)
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

            x1 = int(y * self.move_body_scale + xl1 + xw1 + self.body_pos_screen_wh[0])
            y1 = int(-x * self.move_body_scale + yl1 + yw1 + self.body_pos_screen_wh[1])
            x2 = int(y * self.move_body_scale - xl2 + xw2 + self.body_pos_screen_wh[0])
            y2 = int(-x * self.move_body_scale - yl2 + yw2 + self.body_pos_screen_wh[1])
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
            x1 = int(y * self.move_body_scale + xl1 - xw1 + self.body_pos_screen_wh[0])
            y1 = int(-x * self.move_body_scale + yl1 - yw1 + self.body_pos_screen_wh[1])
            x2 = int(y * self.move_body_scale - xl2 - xw2 + self.body_pos_screen_wh[0])
            y2 = int(-x * self.move_body_scale - yl2 - yw2 + self.body_pos_screen_wh[1])
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)

            x1 = int(y * self.move_body_scale + xw1 + xl1 + self.body_pos_screen_wh[0])
            y1 = int(-x * self.move_body_scale + yw1 + yl1 + self.body_pos_screen_wh[1])
            x2 = int(y * self.move_body_scale - xw2 + xl2 + self.body_pos_screen_wh[0])
            y2 = int(-x * self.move_body_scale - yw2 + yl2 + self.body_pos_screen_wh[1])
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
            x1 = int(y * self.move_body_scale + xw1 - xl1 + self.body_pos_screen_wh[0])
            y1 = int(-x * self.move_body_scale + yw1 - yl1 + self.body_pos_screen_wh[1])
            x2 = int(y * self.move_body_scale - xw2 - xl2 + self.body_pos_screen_wh[0])
            y2 = int(-x * self.move_body_scale - yw2 - yl2 + self.body_pos_screen_wh[1])
            cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255),
                     max(1, int(0.02 * self.move_body_scale)))
            self.update_lidar()

    def go_to_pos(self, x, y):
        """
        configures and sends "go to position" command for robot
        :param x: x position
        :param y: y position
        :return:
        """
        self.target_body_pos = [int(x), int(y), 0]
        x, y = (x - self.body_pos_screen_wh[0]) / self.move_body_scale, (-y + self.body_pos_screen_wh[1]) / self.move_body_scale
        self.robot.go_to(y, x)

    def update_arm(self, scale=1.0):
        """
        updates and draws manipulator data on arm_screen
        :param scale: drawing scale
        :return:
        """
        self.arm_screen = np.copy(self.arm_background)
        m1_ang, m2_ang, m3_ang, m4_ang, m5_ang, grip = *map(math.radians, self.robot.arm_pos[0][:-1]), \
        self.robot.arm_pos[0][
            -1]
        color = (100, 100, 255)

        for i in range(2):
            m2_ang += self.m2_ang_offset
            m3_ang += self.m3_ang_offset
            m4_ang += self.m4_ang_offset

            m3_ang += m2_ang
            m4_ang += m3_ang
            m2x = self.start_point_x
            m2y = self.arm_height - self.start_point_y
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
                m1_ang, m2_ang, m3_ang, m4_ang, m5_ang = map(math.radians, self.robot.arm)
            except:
                break
            color = (255, 255, 255)

    def mouse_on_arm(self, *args):
        """
        service function. Called when mouse is on arm work area.
        Draws target. If mouse pressed changes manipulator target position
        :param args: set: relative mouse position, is mouse pressed
        :return:
        """
        if self.economy_mode:
            return
        pos = args[0]
        pressed = args[1]
        self.target[1] = (self.screen.mouse_wheel_pos / 10 + math.pi / 2) % (2 * math.pi)
        target = [[(pos[0] - self.start_point_x) * self.cylindrical_scale,
                   (-pos[1] + self.arm_height - self.start_point_y) * self.cylindrical_scale], self.target[1]]
        _, _, _, available = self.robot.solve_arm(target)
        if pressed:
            color = ((0, 230, 0) if available else (230, 0, 0))
            self.target[0] = [(pos[0] - self.start_point_x) * self.cylindrical_scale,
                              (-pos[1] + self.arm_height - self.start_point_y) * self.cylindrical_scale]
            self.robot.move_arm(target=self.target)
        else:
            color = ((100, 255, 100) if available else (240, 100, 100))
        m1_ang, m2_ang, m3_ang, m4_ang, m5_ang, grip = self.robot.arm_pos[0]
        self.m1_slider.set_val(m1_ang)
        self.m2_slider.set_val(m2_ang)
        self.m3_slider.set_val(m3_ang)
        self.m4_slider.set_val(m4_ang)
        self.m5_slider.set_val(m5_ang)
        self.grip_slider.set_val(grip)
        cv2.circle(self.arm_screen, (int(pos[0]), int(pos[1])), 10, color, 4)
        size = 20
        xs = int(pos[0] + size * 0.6 * math.cos(-self.target[1] - math.pi / 2))
        ys = int(pos[1] + size * 0.6 * math.sin(-self.target[1] - math.pi / 2))
        xe = int(pos[0] - size * 1.4 * math.cos(-self.target[1] - math.pi / 2))
        ye = int(pos[1] - size * 1.4 * math.sin(-self.target[1] - math.pi / 2))
        cv2.line(self.arm_screen, (xs, ys), (xe, ye), color, 2)

    ###############################----Not working----##############################################
    def draw_reachable(self):
        m2_range = [-63, 84]
        m3_range = [-110, 135]
        m4_range = [-120, 60]
        center = [0, 0]
        center[0] = int(self.start_point_x + self.m4_len * math.sin(self.target[1]) / self.cylindrical_scale)
        center[1] = int(
            self.arm_height - self.start_point_y + self.m4_len * math.cos(self.target[1]) / self.cylindrical_scale)
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
