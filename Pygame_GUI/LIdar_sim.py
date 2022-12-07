from Objects import *
from Screen import Screen
from Slam_test import *
import numpy as np

deb = True


def debug(inf):
    if deb:
        print(inf)


class Lidar_sim:
    def __init__(self, width, inp_file_name):

        # window properties
        self.width = width

        # canvases
        self.body_pos_background = np.array([[[20, 70, 190]] * width] * width, dtype=np.uint8)
        self.body_pos_screen = np.copy(self.body_pos_background)

        self.move_body_scale = 60
        self.wall_lines = []
        self.pause = False
        self.space_clk = False
        self.step = True

        f = open(inp_file_name, "r")
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
        self.odom, self.lidar = self.log_data[self.log_data_ind]

        # for slam:
        self.last_odom = [0, 0, 0]
        self.pose_graph = {0: [[0, 0, 0], None, None, []]}  # odom, object, lidar, children[child id, edge id]]
        self.edges = np.array([])
        self.edge_num = 0
        self.node_num = 0
        self.nodes_coords = np.array(False)

    def init_pygame(self):
        """
        Initialises PyGame and precreated pygame objects:
        two buttons to change camera mode and six sliders to control arm
        """
        self.screen = Screen(self.width, self.width * 1.09)
        self.body_pos_pygame = Mat(self.screen, x=0, y=0, cv_mat_stream=self.body_pos_stream)
        self.clock = pg.time.Clock()

        self.m1_slider = Slider(self.screen,
                                min=0, max=len(self.log_data), val=0,
                                x=0.029, y=0.94,
                                width=0.942, height=0.032,
                                color=(150, 160, 170),
                                func=self.change_ind)

    def body_pos_stream(self):
        """
        service function for correct work with CvMat
        :return: map CvMat
        """
        return self.body_pos_screen

    def change_ind(self, val):
        """
        service function for changing manipulator's joint angle
        :param val: angle of joint 1
        :return:
        """
        self.log_data_ind = int(val)

    def run(self):
        """
        main cycle
        initialises PyGame, updates all data, check pressed keys, updates screen
        :return:
        """
        self.init_pygame()
        while self.screen.running:
            self.update_keys()
            if self.step and not self.pause:
                self.step = False
                self.body_pos_screen = np.copy(self.body_pos_background)
                self.update_body_pos()

                self.odom, self.lidar = self.log_data[self.log_data_ind]

                if self.log_data_ind + 1 < len(self.log_data):
                    self.log_data_ind += 1
                    self.step = True

                self.m1_slider.set_val(self.log_data_ind)
                self.update_lidar()

                self.process_SLAM()

            self.screen.step()
            self.clock.tick(9)

    def homogen_matrix_from_odom(self, odom):
        x, y, rot = odom
        rot = rot - math.pi / 2
        robot_rad = 0.3
        X = np.array([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]])

        f = np.array([[1, 0, 0],
                      [0, 1, robot_rad],
                      [0, 0, 1]])

        rot = np.array([[math.cos(rot), -math.sin(rot), 0],
                        [math.sin(rot), math.cos(rot), 0],
                        [0, 0, 1]])

        X = np.dot(X, np.dot(rot, f))
        return X

    def add_edge(self, from_n, to_n):
        #print("from:", from_n)
        #print("to:", to_n)
        if isinstance(from_n, np.ndarray):
            A = np.linalg.inv(from_n)
        else:
            A = np.linalg.inv(self.homogen_matrix_from_odom(from_n))
        if isinstance(to_n, np.ndarray):
            B = np.copy(to_n)
        else:
            B = self.homogen_matrix_from_odom(to_n)
        if self.edge_num > 0:
            self.edge = np.append(self.edge, [np.dot(A, B)], axis=0)
            self.edge_num += 1
        else:
            self.edge = np.array([np.dot(A, B)])
            self.edge_num += 1
        #print(np.dot(A, B))

    def draw_line_from_point_cloud(self, object, color=(255, 255, 255)):
        approx_points = douglas_peucker(object, True)[0]
        connection_coords = [[int(self.width // 2 + object[p][1]*self.move_body_scale),
                              int(self.width // 2 - object[p][0]*self.move_body_scale)] for p in approx_points]
        #connection_coords = [
        #    [int(p[1]),
        #     int(p[0])]
        #   for p in connection_coords]
        for dot in range(1, len(connection_coords)):
            cv2.line(self.body_pos_screen, connection_coords[dot - 1], connection_coords[dot], color,
                     max(1, int(0.05 * self.move_body_scale)))

    def generate_weight_vector(self, src):
        ind = src[0]
        w = src[1]
        out = []
        print(src)
        curr = 0
        step = w[0] / (ind[1] - ind[0])
        for a in range(0, ind[1] - ind[0]):
            out.append(int(curr))
            curr += step
        for i in range(2, len(ind)-1):
            step = -2*w[i - 2] / (ind[i] - ind[i-1])
            curr = w[i - 2] - step
            for a in range(0, (ind[i] - ind[i-1]) // 2):
                out.append(int(curr))
                curr += step
            curr = 0
            step = 2*w[i-1] / (ind[i] - ind[i-1])
            for a in range(0, (ind[i] - ind[i - 1])//2):
                out.append(int(curr))
                curr += step
            if (ind[i] - ind[i-1]) % 2 == 1:
                out.append(int(curr))
        step = -w[-1] / (ind[-1] - ind[-2])
        curr = w[-1] - step
        for a in range(0, (ind[-1] - ind[-2])):
            out.append(int(curr))
            curr += step
        return out


    def process_SLAM(self):
        object_coords = split_objects(self.log_data[self.log_data_ind])
        object = object_coords[0]
        #scaled_odom = [*scle_to_arr(*self.odom[:2]), self.odom[2]]
        #print(scaled_odom, self.odom)
        self.draw_line_from_point_cloud(object)
        peaks = douglas_peucker(object, True)
        corners, corner_lines = find_corners(object, peaks[0])
        x, y, r = self.odom
        xo, yo, ro = self.last_odom
        if math.sqrt((x - xo) ** 2 + (y - yo) ** 2) > 0.01 or abs(r - ro) > 0.5 or self.node_num == 0:
            if self.node_num == 0:
                self.nodes_coords = (np.array([self.odom]))
                self.pose_graph[self.node_num] = [self.odom, None, self.lidar, []]
            elif not corners:
                self.pose_graph[self.node_num] = [self.odom, None, self.lidar, []]
                self.pose_graph[self.node_num - 1][3].append([self.node_num, self.edge_num])
                self.add_edge(self.pose_graph[self.node_num - 1][0], self.odom)
                #pyplot.plot(*zip(self.odom[:2]), 'o', label=f'robot1- ang: {self.odom[2]}')
                #pyplot.plot(*zip(self.pose_graph[self.node_num - 1][0][:2]), 'o', label=f'robot2- ang: {self.pose_graph[self.node_num - 1][0][2]}')
                #pyplot.axis('equal')
                #pyplot.legend(numpoints=1)
                #pyplot.show()
            else:
                corners = sorted(corners, key=lambda x: x[1], reverse=True)

                #object = np.array(corners[0][0])
                object = np.array(object_coords[0])
                self.pose_graph[self.node_num] = [self.odom, object, self.lidar, []]
                _, cl_points = self.find_n_closest(self.odom, min(5, self.node_num))
                self.nodes_coords = np.append(self.nodes_coords, [self.odom], axis=0)
                if isinstance(cl_points, int):
                    cl_points = [cl_points]
                for nn in cl_points:
                    if isinstance(self.pose_graph[nn][1], np.ndarray):
                        weight_vector = self.generate_weight_vector(peaks)
                        print(len(weight_vector))
                        print(len(object[peaks[0][0]:peaks[0][-1]]))
                        icp_out = icp(object[peaks[0][0]:peaks[0][-1]], np.array(self.pose_graph[nn][1]), weight_vector=[weight_vector])
                        # draw icp
                        if icp_out[-1] and icp_out[-1] < 0.1:
                            print(icp_out[-1])
                            self.pose_graph[nn][3].append([self.node_num, self.edge_num])
                            mat_odom = self.homogen_matrix_from_odom(self.odom)
                            T = np.copy(icp_out[0])
                            #print("T:", T)
                            mat_odom_rot = mat_odom[:2, :2]
                            mat_odom_t = mat_odom[:2, 2]
                            T_rot = T[:2, :2]
                            T_t = T[:2, 2] / T[2, 2]
                            mat_odom_rot = np.dot(mat_odom_rot, T_rot)
                            mat_odom_t = mat_odom_t + T_t
                            corrected_odom = np.eye(3)
                            corrected_odom[:2, :2] = mat_odom_rot
                            corrected_odom[:2, 2] = T_t


                            #pyplot.figure(1)
                            #pyplot.subplot()

                            #pyplot.plot(*zip(self.odom[:2]), 'o', label=f'robot1- ang: {self.odom[2]}')
                            #pyplot.plot(*zip(corrected_odom[:2, 2]), 'o', label=f'robot_c- ang: {math.acos(corrected_odom[0, 0])}')
                            #pyplot.plot(*zip(self.pose_graph[nn][0][:2]), 'o',
                            #            label=f'robot2- ang: {self.pose_graph[nn][0][2]}')
                            #pyplot.axis('equal')
                            #pyplot.legend(numpoints=1)


                            self.add_edge(self.pose_graph[nn][0], corrected_odom)
                            #self.draw_line_from_point_cloud(object, (0, 255, 0))
                            #self.draw_line_from_point_cloud(self.pose_graph[nn][1], (0, 0, 255))
                            self.screen.step()
                            self.clock.tick(6)


                            pyplot.figure(2)
                            pyplot.subplot()
                            pyplot.plot([p[0] for p in object], [p[1] for p in object], 'o', label='points 2')

                            # to homogeneous
                            converted = np.ones((object.shape[1] + 1, object.shape[0]))
                            converted[:object.shape[1], :] = np.copy(object.T)
                            # transform
                            converted = np.dot(icp_out[0], converted)
                            # back from homogeneous to cartesian
                            converted = np.array(converted[:converted.shape[1], :]).T
                            pyplot.plot([p[0] for p in converted], [p[1] for p in converted], 'o', label='converted')
                            pyplot.plot([p[0] for p in self.pose_graph[nn][1]], [p[1] for p in self.pose_graph[nn][1]],
                                        '.', label='points 1')

                            pyplot.axis('equal')
                            pyplot.legend(numpoints=1)
                            pyplot.show()


            self.last_odom = self.odom
            self.node_num += 1

    def find_n_closest(self, point, n):
        node_arr = self.nodes_coords
        if not node_arr.any():
            return None, None
        nodes_tree = scipy.spatial.cKDTree(node_arr)
        return nodes_tree.query(point, n)

    def update_keys(self):
        """
        checks pressed keys and configure commands to send according to pressed keys
        :return:
        """
        pressed_keys = self.screen.pressed_keys
        if pg.K_SPACE in pressed_keys:
            if not self.space_clk:
                self.pause = not self.pause
                self.space_clk = True
        elif pg.K_SPACE not in pressed_keys:
            self.space_clk = False
        if pg.K_LEFT in pressed_keys:
            self.log_data_ind -= 1
            self.step = True
        if pg.K_RIGHT in pressed_keys:
            self.log_data_ind += 1
            self.step = True

    def draw_P_TH_line(self, P, TH, color=(255, 255, 0)):
        w = self.width // 2 * discrete
        x1, y1, x2, y2, x3, y3, x4, y4 = w, w, w, w, w, w, w, w
        cTH = math.cos(TH)
        sTH = math.sin(TH)
        y1 = w
        y3 = -w
        if math.sin(TH):
            x1 = int((P - y1 * cTH) / sTH)
            x3 = int((P - y3 * cTH) / sTH)
        x2 = w
        x4 = -w
        if cTH:
            y2 = int((P - x2 * sTH) / cTH)
            y4 = int((P - x4 * sTH) / cTH)
        k = discrete / self.move_body_scale
        cross = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        line = []
        for i in cross:
            if -w <= i[0] <= w and -w <= i[1] <= w:
                line.append((int(i[0] // k) + self.width // 2, int(i[1] // k + self.width // 2)))
        if len(line) > 1:
            cv2.line(self.body_pos_screen, line[0], line[1], color, max(1, int(0.05 * self.move_body_scale)))



    def update_lidar(self):
        """
        draws lidar data on body_pos_screen
        :return:
        """
        odom = self.odom
        lidar = self.lidar
        x, y, ang = odom
        if lidar:
            cent_y, cent_x = y * self.move_body_scale + self.width // 2, -x * self.move_body_scale + self.width // 2
            cent_y = int(cent_y - 0.3 * self.move_body_scale * math.cos(ang + math.pi / 2))
            cent_x = int(cent_x - 0.3 * self.move_body_scale * math.sin(ang + math.pi / 2))
            for l in range(0, len(lidar), 1):
                if not 0.01 < lidar[l] < 5.5:
                    continue
                color = (0, max(255, 255 - int(45.5 * l)), min(255, int(45.5 * l)))
                color = (20, 90, 210)
                cv2.ellipse(self.body_pos_screen, (cent_y, cent_x),
                            (int(lidar[l] * self.move_body_scale), int(lidar[l] * self.move_body_scale)),
                            math.degrees(ang), 30 + int(-240 / len(lidar) * l), 30 + int(-240 / len(lidar) * (l + 1)),
                            color,
                            max(1, int(0.1 * self.move_body_scale)))

    def update_body_pos(self, *args):
        """
        draws body rectangle on body_pos_screen and sends robot to set position if mouse pressed
        :param args: set: relative mouse position and is mouse pressed
        :return:
        """

        odom = self.odom
        x, y, ang = odom
        cv2.circle(self.body_pos_screen,
                   (int(y * self.move_body_scale + self.width // 2), int(-x * self.move_body_scale + self.width // 2)),
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

        x1 = int(y * self.move_body_scale + xl1 + xw1 + self.width // 2)
        y1 = int(-x * self.move_body_scale + yl1 + yw1 + self.width // 2)
        x2 = int(y * self.move_body_scale - xl2 + xw2 + self.width // 2)
        y2 = int(-x * self.move_body_scale - yl2 + yw2 + self.width // 2)
        cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
        x1 = int(y * self.move_body_scale + xl1 - xw1 + self.width // 2)
        y1 = int(-x * self.move_body_scale + yl1 - yw1 + self.width // 2)
        x2 = int(y * self.move_body_scale - xl2 - xw2 + self.width // 2)
        y2 = int(-x * self.move_body_scale - yl2 - yw2 + self.width // 2)
        cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)

        x1 = int(y * self.move_body_scale + xw1 + xl1 + self.width // 2)
        y1 = int(-x * self.move_body_scale + yw1 + yl1 + self.width // 2)
        x2 = int(y * self.move_body_scale - xw2 + xl2 + self.width // 2)
        y2 = int(-x * self.move_body_scale - yw2 + yl2 + self.width // 2)
        cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
        x1 = int(y * self.move_body_scale + xw1 - xl1 + self.width // 2)
        y1 = int(-x * self.move_body_scale + yw1 - yl1 + self.width // 2)
        x2 = int(y * self.move_body_scale - xw2 - xl2 + self.width // 2)
        y2 = int(-x * self.move_body_scale - yw2 - yl2 + self.width // 2)
        cv2.line(self.body_pos_screen, (x1, y1), (x2, y2), (255, 255, 255),
                 max(1, int(0.02 * self.move_body_scale)))


sim = Lidar_sim(700, "../lidar_odom_log/lidar_odom_log_9.txt")
sim.run()
