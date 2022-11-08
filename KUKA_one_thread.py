import io
import math
import socket
import threading as thr
import time

import cv2
import numpy as np
import paramiko
from PIL import Image
from mjpeg.client import MJPEGClient

deb = True


def debug(inf):
    if deb:
        print(inf)


def range_cut(mi, ma, val):
    return min(ma, max(mi, val))


class KUKA:
    """
    KUKA youbot controller
    """
    def __init__(self, ip, /, ros=False, offline=False, read_depth=True):
        """Initializes robot KUKA youbot
        Establishes connection with depth and RGB video server
        Establishes connection with control and sensor socket
        Lunches

        :param ip: robot ip
        :param ros: automatically lunches youbot_tl_tests on KUKA if true
        :param offline: toggles offline mode (doesn't try to connect to robot)
        :param read_depth: if false doesn't start depth client
        """
        self.operating = True
        self.ip = ip
        self.frequency = 120
        self.send_time = 0
        self.send_queue = [None, None, None, None]

        # camera receive buffer
        self.data_buff = b''
        self.max_buff_len = 1500
        self.cam_image = np.array([[[190, 70, 20]] * 640] * 480, dtype=np.uint8)
        self.cam_image_BGR = np.array([[[20, 70, 190]] * 640] * 480, dtype=np.uint8)
        self.cam_depth = np.array([[[190, 70, 20]] * 640] * 480, dtype=np.uint8)

        # control
        self.arm_pos = [0, 56, -80, -90, 0, 2]
        self.body_target_pos = [0, 0, 0]
        self.going_to_target_pos = False
        self.move_speed = (0, 0, 0)
        self.move_to_target_max_speed = 2
        self.move_to_target_k = 1

        # dimensions
        self.m2_len = 155
        self.m3_len = 135
        self.m4_len = 200

        # sensor data
        self.lidar_data = None
        self.increment_data = None
        self.increment_data_lidar = None
        self.corr_arm_pos = None

        # connection
        debug(f"connecting to {ip}")
        self.connected = True
        if offline:
            self.connected = False
            return
        if ros:
            debug("starting ROS")
            self.connect_ssh()
            time.sleep(5)
            debug("ROS launched")
        try:
            debug("connecting to control channel")
            # init socket
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.settimeout(2)
            self.conn.connect((self.ip, 7777))

            # init reading thread
            self.data_lock = thr.Lock()
            self.send_lock = thr.Lock()
            self.data_thr = thr.Thread(target=self._receive_data, args=())
            self.send_thr = thr.Thread(target=self.send_data, args=())
            time.sleep(1)
            self.data_thr.start()
            self.send_thr.start()

            debug("connected *maybe* to 7777 (data stream)")

        except:
            self.connected *= False
            debug("failed to connect to 7777 (data stream)")

        if self.connected:
            if read_depth:
                self.init_depth_client()
            self.init_rgb_client()

        self.operating = self.connected
        if self.connected:
            while not self.corr_arm_pos:
                time.sleep(0.1)
            self.arm_pos[:-1] = self.corr_arm_pos

    def init_rgb_client(self):
        debug("connecting to video channel")
        self.client_rgb = MJPEGClient(
            f"http://{self.ip}:8080/stream?topic=/camera/rgb/image_rect_color&width=640&height=480&quality=20")
        bufs = self.client_rgb.request_buffers(65536, 5)
        for b in bufs:
            self.client_rgb.enqueue_buffer(b)
        self.client_rgb.start()
        self.cam_rgb_lock = thr.Lock()
        self.cam_rgb_thr = thr.Thread(target=self.get_frame_rgb, args=())
        self.cam_rgb_thr.start()

    def init_depth_client(self):
        self.client_depth = MJPEGClient(
            f"http://192.168.88.25:8080/stream?topic=/camera/depth/image_rect")
        bufsd = self.client_depth.request_buffers(65536, 5)
        for b in bufsd:
            self.client_depth.enqueue_buffer(b)
        self.client_depth.start()
        self.cam_depth_lock = thr.Lock()
        self.cam_depth_thr = thr.Thread(target=self.get_frame_depth, args=())
        self.cam_depth_thr.start()

    def connect_ssh(self):
        user = 'root'
        secret = '111111'
        port = 22
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=self.ip, username=user, password=secret, port=port)
        ssh = client.invoke_shell()
        time.sleep(0.5)
        ssh.send(b"screen -S roslaunch\n")
        time.sleep(2)
        ssh.send(b"roslaunch youbot_tl_test ytl.launch\n")
        debug(ssh.recv(3000)[-100:].decode("utf-8"))
        client.close()

    # receiving and parsing sensor data

    def post_to_send_data(self, ind, data):
        if self.connected:
            self.send_lock.acquire()
            self.send_queue[ind] = data
            self.send_lock.release()
        else:
            self.send_queue[ind] = data

    def send_data(self):
        send_ind = 0
        while self.operating:
            to_send = None
            while not to_send and self.operating:
                self.send_lock.acquire()
                to_send = self.send_queue[send_ind]
                self.send_queue[send_ind] = None
                self.send_lock.release()
                send_ind += 1
                self.send_time = time.time_ns()
                if send_ind == 4:
                    send_ind = 0
                time.sleep(1 / self.frequency)
            if self.connected:
                self.conn.send(to_send)
            else:
                debug(f"message:{to_send}")
                pass

    def _parse_data(self, data):
        write_lidar = None
        write_increment = None
        write_arm = None
        if data[:7] == ".laser#":
            raw = list(data[7:].split(';'))
            if len(raw) < 160:
                write_lidar = None
            else:
                write_lidar = []
                for i in raw:
                    try:
                        write_lidar.append(float(i))
                    except:
                        if i != "":
                            write_lidar.append(5)

        elif data[:6] == ".odom#":
            try:
                write_increment = list(map(float, data[6:].split(';')))
            except:
                write_increment = None
        elif data[:7] == ".manip#":
            try:
                write_arm = list(map(float, data[7:].split(';')))
            except:
                write_arm = None
        # update data
        if write_lidar or write_increment or write_arm:
            self.data_lock.acquire()
            if write_lidar:
                self.lidar_data = write_lidar
                self.increment_data_lidar = self.increment_data
            if write_increment:
                self.increment_data = write_increment
            if write_arm:
                m1 = -write_arm[0] + 168
                m2 = -write_arm[1] + 66
                m3 = -write_arm[2] - 150
                m4 = -write_arm[3] + 105
                m5 = write_arm[4] - 166
                self.corr_arm_pos = [m1, m2, m3, m4, m5]
            self.data_lock.release()

    def _receive_data(self):
        data_buff_len = 0
        self.data_buff = b''
        while self.operating:
            if data_buff_len > 2000:
                self.data_buff = b''
                data_buff_len = 0
            self.data_buff += self.conn.recv(1)
            data_buff_len += 1
            if self.data_buff[-1] == 13:
                self.conn.recv(1)
                str_data = str(self.data_buff[:-1], encoding='utf-8')
                self.data_buff = b''
                self.data_parser_tht = thr.Thread(target=self._parse_data, args=(str_data,))
                self.data_parser_tht.start()

    # get functions
    @property
    def lidar(self):
        if self.connected:
            self.data_lock.acquire()
            out = self.lidar_data
            inc = self.increment_data_lidar
            self.data_lock.release()
            return inc, out
        else:
            return None, None

    @property
    def arm(self):
        if self.connected:
            self.data_lock.acquire()
            out = self.corr_arm_pos
            self.data_lock.release()
            return out
        else:
            return None, None, None, None, None

    @property
    def increment(self):
        if self.connected:
            self.data_lock.acquire()
            out = self.increment_data
            self.data_lock.release()
            return out
        else:
            return None

    @property
    def cam(self):
        if self.connected:
            self.cam_rgb_lock.acquire()
            out = self.cam_image
            self.cam_rgb_lock.release()
            return out
        else:
            return self.cam_image

    @property
    def cam_BGR(self):
        if self.connected:
            self.cam_rgb_lock.acquire()
            out = self.cam_image_BGR
            self.cam_rgb_lock.release()
            return out
        else:
            return self.cam_image_BGR

    @property
    def depth(self):
        if self.connected:
            self.cam_depth_lock.acquire()
            out = self.cam_depth
            self.cam_depth_lock.release()
            return out
        else:
            return self.cam_depth

    # control base and arm
    # go with set speed
    def move_base(self, f=0.0, s=0.0, r=0.0):
        f = range_cut(-1, 1, f)
        s = range_cut(-1, 1, s)
        r = range_cut(-1, 1, r)
        self.post_to_send_data(0, bytes(f'/base:{f};{s};{r}^^^', encoding='utf-8'))
        self.move_speed = (f, s, r)

    # go to set coordinates
    def go_to(self, x, y, ang=0):
        if self.operating:
            if self.going_to_target_pos:
                self.body_target_pos_lock.acquire()
                self.body_target_pos = [x, y, ang]
                self.body_target_pos_lock.release()
            else:
                self.body_target_pos_lock = thr.Lock()
                self.going_to_target_pos = True
                self.body_target_pos = [x, y, ang]
                self.go_to_tr = thr.Thread(target=self.move_base_to_pos, args=())
                self.go_to_tr.start()

    def move_base_to_pos(self):
        while self.operating and self.going_to_target_pos:
            self.body_target_pos_lock.acquire()
            x, y, ang = self.body_target_pos
            self.body_target_pos_lock.release()
            inc = self.increment
            loc_x = x - inc[0]
            loc_y = y - inc[1]
            rob_ang = inc[2]
            dist = math.sqrt(loc_x ** 2 + loc_y ** 2)
            speed = min(self.move_to_target_max_speed, dist * self.move_to_target_k)
            if dist < 0.005:
                self.move_base(0, 0, 0)
                time.sleep(0.01)
                self.move_base(0, 0, 0)
                self.going_to_target_pos = False
                break
            targ_ang = math.atan2(loc_y, loc_x)
            loc_ang = targ_ang - rob_ang
            fov_speed = speed * math.cos(loc_ang)
            side_speed = -speed * math.sin(loc_ang)
            self.move_base(fov_speed, side_speed, 0)
            time.sleep(1 / self.frequency)
            dist_old = dist

    # move arm forward or inverse kinetic
    def move_arm(self, *args, **kwargs):
        grab = None
        self.arm_moving = False
        if args:
            self.arm_pos[:len(args)] = args
            if len(args) == 6:
                grab = True
        if list(kwargs.keys()).count("m1") > 0:
            self.arm_pos[0] = kwargs["m1"]
        if list(kwargs.keys()).count("m2") > 0:
            self.arm_pos[1] = kwargs["m2"]
        if list(kwargs.keys()).count("m3") > 0:
            self.arm_pos[2] = kwargs["m3"]
        if list(kwargs.keys()).count("m4") > 0:
            self.arm_pos[3] = kwargs["m4"]
        if list(kwargs.keys()).count("m5") > 0:
            self.arm_pos[4] = kwargs["m5"]
        if list(kwargs.keys()).count("grab") > 0:
            self.arm_pos[5] = kwargs["grab"]
            self.post_to_send_data(3, bytes(f'/grip:0;{self.arm_pos[5]}^^^', encoding='utf-8'))

        if list(kwargs.keys()).count("target") > 0:
            if len(kwargs["target"][0]) == 2:
                m2, m3, m4, _ = self.solve_arm(kwargs["target"])
                self.arm_pos[1:4] = m2, m3, m4

        m1 = range_cut(11, 302, -self.arm_pos[0] + 168)
        m2 = range_cut(3, 150, -self.arm_pos[1] + 66)
        m3 = range_cut(-260, -15, -self.arm_pos[2] - 150)
        m4 = range_cut(10, 195, -self.arm_pos[3] + 105)
        m5 = range_cut(21, 292, self.arm_pos[4] + 166)
        self.post_to_send_data(1, bytes(f'/arm:0;{m1};{m2};{m3};{m4};{m5}^^^', encoding='utf-8'))

    # solve inverse kinetic

    def solve_arm(self, target, cartesian=False):
        if not cartesian:
            x = target[0][0]
            y = target[0][1]
            ang = target[1]
            try:
                x -= self.m4_len * math.sin(ang)
                y += self.m4_len * math.cos(ang)
                fi = math.atan2(y, x)
                b = math.acos((self.m2_len ** 2 + self.m3_len ** 2 - x ** 2 - y ** 2) / (2 * self.m2_len * self.m3_len))
                a = math.acos((self.m2_len ** 2 - self.m3_len ** 2 + x ** 2 + y ** 2) / (
                        2 * self.m2_len * math.sqrt(x ** 2 + y ** 2)))
                m2_ang = fi + a - math.pi / 2
                m3_ang = b - math.pi
                m4_ang = (ang - m2_ang - m3_ang - math.pi)
                m2_ang_neg = fi - a - 3 * math.pi / 2 + math.pi
                m3_ang_neg = -b + math.pi
                m4_ang_neg = (ang + a + b - fi - 3 * math.pi / 2)
                m2_ang, m3_ang, m4_ang = map(math.degrees, (m2_ang, m3_ang, m4_ang))
                m2_ang_neg, m3_ang_neg, m4_ang_neg = map(math.degrees, (m2_ang_neg, m3_ang_neg, m4_ang_neg))
                if -84 < m2_ang < 63 and -135 < m3_ang < 110 and -90 < m4_ang < 95:
                    return m2_ang, m3_ang, m4_ang, True
                elif -84 < m2_ang_neg < 63 and -135 < m3_ang_neg < 110 and -90 < m4_ang_neg < 95:
                    return m2_ang_neg, m3_ang_neg, m4_ang_neg, True
                else:
                    return *self.arm_pos[1:4], False
                # m2_ang = range_cut(-84, 63, m2_ang)
                # m3_ang = range_cut(-135, 110, m3_ang)
                # m4_ang = range_cut(-120, 90, m4_ang)
            except:
                # debug("math error, out of range")
                return *self.arm_pos[1:4], False
        else:
            x = target[0][0]
            y = target[0][1]
            z = target[0][2]
            ang = target[1]
            xy = math.sqrt(x ** 2 + y ** 2)
            try:
                self.arm_pos[0] = math.degrees(math.asin(x / xy))
                z -= self.m4_len * math.sin(ang)
                xy += self.m4_len * math.cos(ang)
                fi = math.atan2(z, xy)
                b = math.acos(
                    (self.m2_len ** 2 + self.m3_len ** 2 - xy ** 2 - z ** 2) / (2 * self.m2_len * self.m3_len))
                a = math.acos((self.m2_len ** 2 - self.m3_len ** 2 + xy ** 2 + z ** 2) / (
                        2 * self.m2_len * math.sqrt(xy ** 2 + z ** 2)))
                m2_ang = fi + a - 3 * math.pi / 2 + math.pi
                m3_ang = b - math.pi
                return list(map(math.degrees, (m2_ang, m3_ang, ang - a - b - fi + math.pi / 2)))
            except:
                debug("math error, out of range")
                return self.arm_pos[1:4]

    # video capture

    def get_frame_rgb(self):
        while self.operating:
            buf_rgb = self.client_rgb.dequeue_buffer()
            image = Image.open(io.BytesIO(buf_rgb.data))
            imageBGR = np.array(image)

            imageRGB = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            self.client_rgb.enqueue_buffer(buf_rgb)

            self.cam_rgb_lock.acquire()
            self.cam_image = imageRGB
            self.cam_image_BGR = imageBGR
            self.cam_rgb_lock.release()

    def get_frame_depth(self):
        while self.operating:
            buf_depth = self.client_depth.dequeue_buffer()
            image_depth = np.array(Image.open(io.BytesIO(buf_depth.data)))
            # image_depth = cv2.cvtColor(np.array(image_depth), cv2.COLOR_BGR2GRAY)
            self.client_depth.enqueue_buffer(buf_depth)

            self.cam_depth_lock.acquire()
            self.cam_depth = image_depth
            self.cam_depth_lock.release()

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self.connected:
            self.operating = False
            self.move_base()
            self.move_arm(0, 56, -80, -90, 0, 2)
            time.sleep(1)
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
        else:
            debug("disconnected")
