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


class KUKA():
    def __init__(self, ip='192.168.88.25', **kwargs):
        self.operating = True
        self.ip = ip
        self.frequency = 200
        self.send_time = 0

        # camera receive buffer
        self.data_buff = b''
        self.max_buff_len = 1500
        self.cam_image = np.array([[[190, 70, 20]] * 680] * 480, dtype=np.uint8)

        # connection
        debug(f"connecting to {ip}")
        self.connected = True
        if list(kwargs.keys()).count("ros") > 0 and kwargs["ros"]:
            debug("starting ROS")
            self.connect_ssh()
            time.sleep(10)
            debug("ROS launched")
        try:
            debug("connecting to control channel")
            # init socket
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.settimeout(3)
            self.conn.connect((self.ip, 7777))

            # init reading thread
            self.data_lock = thr.Lock()
            self.data_thr = thr.Thread(target=self._receive_data, args=())
            time.sleep(1)
            self.data_thr.start()
            debug("connected *maybe* to 7777 (data stream)")

        except:
            self.connected *= False
            debug("failed to connect to 7777 (data stream)")

        if self.connected:
            debug("connecting to video channel")
            self.client = MJPEGClient(
                f"http://{self.ip}:8080/stream?topic=/camera/rgb/image_rect_color&width=640&height=480&quality=50")
            bufs = self.client.request_buffers(65536, 50)
            for b in bufs:
                self.client.enqueue_buffer(b)
            self.client.start()
            self.cam_lock = thr.Lock()
            self.cam_thr = thr.Thread(target=self.get_frame, args=())
            self.cam_thr.start()

        self.operating = self.connected

        # control
        self.arm_pos = [0, 56, -80, -90, 0, 2]
        self.body_target_pos = [0, 0, 0]
        self.going_to_target_pos = False
        self.move_speed = (0, 0, 0)
        self.move_to_target_speed = 0.2
        self.move_to_target_correction_speed = 0.03

        # dimensions
        self.m2_len = 155
        self.m3_len = 135
        self.m4_len = 200

        # sensor data
        self.lidar_data = None
        self.increment_data = None
        self.increment_data_lidar = None

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
        ssh.send(b"roslaunch youbot_tactical_level ytl.launch\n")
        debug(ssh.recv(3000)[-100:].decode("utf-8"))
        client.close()

    # receiving and parsing sensor data

    def send_data(self, data):
        if self.connected:
            if time.time_ns() - self.send_time > 1000000000//self.frequency:
                self.send_time = time.time_ns()
                self.conn.send(data)
        else:
            # debug(f"message:{data}")
            pass

    def _parse_data(self, data):
        write_lidar = None
        write_increment = None

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
        # update data
        if write_lidar or write_increment:
            self.data_lock.acquire()
            if write_lidar:
                self.lidar_data = write_lidar
                self.increment_data_lidar = self.increment_data
            if write_increment:
                self.increment_data = write_increment
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
            self.cam_lock.acquire()
            out = self.cam_image
            self.cam_lock.release()
            return out
        else:
            return self.cam_image


    # control base and arm
    # go with set speed
    def move_base(self, f=0.0, s=0.0, r=0.0):
        f = range_cut(-1, 1, f)
        s = range_cut(-1, 1, s)
        r = range_cut(-1, 1, r)
        self.send_data(bytes(f'LUA_Base({f},{s},{r})^^^', encoding='utf-8'))
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
        dist_old = 0
        state = 0
        while self.operating and self.going_to_target_pos:
            self.body_target_pos_lock.acquire()
            x, y, ang = self.body_target_pos
            self.body_target_pos_lock.release()
            inc = self.increment
            loc_x = x - inc[0]
            loc_y = y - inc[1]
            rob_ang = inc[2]
            dist = math.sqrt(loc_x ** 2 + loc_y ** 2)

            if dist < 0.03:
                speed = self.move_to_target_correction_speed
                state = 3 if state == 2 else state
            if dist < 0.07:
                speed = self.move_to_target_correction_speed
                state = 1 if state == 0 else state
            else:
                speed = self.move_to_target_speed
                state = 0
            if dist < 0.005:
                self.move_base(0, 0, 0)
                time.sleep(0.01)
                self.move_base(0, 0, 0)
                self.going_to_target_pos = False
                break
            elif dist > dist_old or state in [1, 3]:
                targ_ang = math.atan2(loc_y, loc_x)
                loc_ang = targ_ang - rob_ang
                fov_speed = speed * math.cos(loc_ang)
                side_speed = -speed * math.sin(loc_ang)
                self.move_base(fov_speed, side_speed, 0)
                state = 4 if state == 3 else (2 if state == 1 else 0)

            time.sleep(0.0005)
            dist_old = dist

    # move arm forward or inverse kinetic
    def move_arm(self, *args, **kwargs):
        grab = None
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
            grab = True

        if list(kwargs.keys()).count("target") > 0:
            if len(kwargs["target"][0]) == 2:
                m2, m3, m4, _ = self.solve_arm(kwargs["target"])
                self.arm_pos[1:4] = [m2, m3, m4]

        m1 = range_cut(11, 302, -self.arm_pos[0] + 168)
        m2 = range_cut(3, 150, -self.arm_pos[1] + 66)
        m3 = range_cut(-260, -15, -self.arm_pos[2] - 150)
        m4 = range_cut(10, 195, -self.arm_pos[3] + 105)
        m5 = range_cut(21, 292, self.arm_pos[4] + 166)

        self.send_data(bytes(f'LUA_ManipDeg(0, {m1}, {m2}, {m3}, {m4}, {m5})^^^', encoding='utf-8'))
        if grab is not None:
            self.send_data(bytes(f'LUA_Gripper(0, {self.arm_pos[5]})^^^', encoding='utf-8'))

    def grab(self, grab):
        grab = range_cut(0, 2, grab)
        self.arm_pos[-1] = 0
        self.send_data(bytes(f'LUA_Gripper(0, {grab})^^^', encoding='utf-8'))

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
                m2_ang = fi + a - 3 * math.pi / 2 + math.pi
                m3_ang = b - math.pi
                m4_ang = (ang - a - b - fi + math.pi / 2)
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

    def get_frame(self):
        while self.operating:
            buf = self.client.dequeue_buffer()
            image = Image.open(io.BytesIO(buf.data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            self.cam_lock.acquire()
            self.cam_image = image
            self.cam_lock.release()
            self.client.enqueue_buffer(buf)

    def disconnect(self):
        if self.connected:
            self.operating = False
            self.move_base()
            self.move_arm(0, 56, -80, -90, 0, 2)
            time.sleep(3)
            self.send_data(b'#end#^^^')
            time.sleep(1)
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
        else:
            debug("disconnected")
