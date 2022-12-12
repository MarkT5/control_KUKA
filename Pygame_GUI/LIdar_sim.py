from Objects import *
from Screen import Screen
from Slam_test import *
import numpy as np
from pose_graph import PoseGrah

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
        self.pose_graph = PoseGrah()  # odom, object, lidar, children[child id, edge id]]
        self.max_Gauss_Newton_iter = 30

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

                self.m1_slider.set_val(self.log_data_ind)
                self.update_lidar()
                self.process_SLAM()

                if self.log_data_ind + 1 < len(self.log_data):
                    self.log_data_ind += 1
                    self.step = True
                else:
                    self.Gauss_Newton()

            self.screen.step()
            # self.clock.tick(1000)

    def view(self, from_n, to_n):
        A = np.linalg.inv(from_n)
        B = np.copy(to_n)
        return np.dot(A, B)

    def draw_line_from_point_cloud(self, object, color=(255, 255, 255)):
        approx_points = douglas_peucker(object, True)[0]
        connection_coords = [[int(self.width // 2 + object[p][1] * self.move_body_scale),
                              int(self.width // 2 - object[p][0] * self.move_body_scale)] for p in approx_points]
        for dot in range(1, len(connection_coords)):
            cv2.line(self.body_pos_screen, connection_coords[dot - 1], connection_coords[dot], color,
                     max(1, int(0.05 * self.move_body_scale)))

    def generate_weight_vector(self, src):
        ind = src[0]
        w = src[1]
        out = []
        curr = 0
        step = w[0] / (ind[1] - ind[0])
        for a in range(0, ind[1] - ind[0]):
            out.append(int(curr))
            curr += step
        for i in range(2, len(ind) - 1):
            step = -2 * w[i - 2] / (ind[i] - ind[i - 1])
            curr = w[i - 2] - step
            for a in range(0, (ind[i] - ind[i - 1]) // 2):
                out.append(int(curr))
                curr += step
            curr = 0
            step = 2 * w[i - 1] / (ind[i] - ind[i - 1])
            for a in range(0, (ind[i] - ind[i - 1]) // 2):
                out.append(int(curr))
                curr += step
            if (ind[i] - ind[i - 1]) % 2 == 1:
                out.append(int(curr))
        step = -w[-1] / (ind[-1] - ind[-2])
        curr = w[-1] - step
        for a in range(0, (ind[-1] - ind[-2])):
            out.append(int(curr))
            curr += step
        return out

    def process_SLAM(self):
        object_coords = split_objects(self.log_data[self.log_data_ind])
        if object_coords:
            object = object_coords[0]
        else:
            return
        self.draw_line_from_point_cloud(object)
        peaks = douglas_peucker(object, True)
        x, y, r = self.odom
        xo, yo, ro = self.last_odom
        if math.sqrt((x - xo) ** 2 + (y - yo) ** 2) > 0.1 or abs(r - ro) > 0.2 or len(self.pose_graph) == 0:
            self.pose_graph.add_node(-1, self.odom, None, self.lidar)
            if len(peaks[0]) > 3 and len(self.pose_graph) > 2:
                object = np.array(object_coords[0])
                _, cl_points = self.pose_graph.find_n_closest(-1, 5)
                if isinstance(cl_points, int):
                    cl_points = [cl_points]
                self.pose_graph[-1, "object"] = object
                for nn in cl_points:
                    if isinstance(self.pose_graph[nn, "object"], np.ndarray):
                        weight_vector = self.generate_weight_vector(peaks)
                        icp_out = icp(object[peaks[0][0]:peaks[0][-1]], np.array(self.pose_graph[nn, "object"]),
                                      weight_vector=[weight_vector])
                        # draw icp
                        if icp_out[-1] and icp_out[-1] < 0.2:
                            corrected_odom = np.dot(icp_out[0], self.pose_graph[-1, "mat_pos"])

                            self.pose_graph[nn, "edge", -1] = corrected_odom  # [parent, "edge", child] = value

                            #object_to_draw = split_objects([self.odom, self.lidar])[0]
                            #pyplot.plot([p[0] for p in object_to_draw], [p[1] for p in object_to_draw], 'o',
                            #           label='points 2')
                            #aco = math.acos(corrected_odom[0, 0])
                            #asi = math.asin(corrected_odom[1, 0])
                            #c_odom = [*corrected_odom[:2, 2], aco * (asi / abs(asi))]
                            #converted = split_objects([corrected_odom, self.lidar])[0]
                            #pyplot.plot([p[0] for p in converted], [p[1] for p in converted], 'o', label='converted')
                            #src = split_objects([self.pose_graph[nn, "mat_pos"], self.pose_graph[nn, "lidar"]])[0]
                            #pyplot.plot([p[0] for p in src], [p[1] for p in src], '.', label='src')
                            #pyplot.axis('equal')
                            #pyplot.legend(numpoints=1)
                            #pyplot.show()

            self.last_odom = self.odom

    def error_func(self, i, Xi, j, Xj):
        Zij = self.pose_graph.edge(i, j)
        Xi = self.pose_graph.homogen_matrix_from_pos(Xi)
        Xj = self.pose_graph.homogen_matrix_from_pos(Xj)

        err_mat = np.linalg.inv(Zij) @ (np.linalg.inv(Xi) @ Xj)
        err = self.pose_graph.pos_vector_from_homogen_matrix(err_mat)
        return err

    def Jacobian(self, i, j):
        eps = 1e-15
        grad_i = []
        grad_j = []

        Xi = self.pose_graph[i, "pos"]
        Xj = self.pose_graph[j, "pos"]

        for a in range(3):
            t = np.zeros(3).astype(float)
            t[a] = t[a] + eps
            grad_i.append((self.error_func(i, Xi + t, j, Xj) - self.error_func(i, Xi - t, j, Xj)) / (2 * eps))
        for b in range(3):
            t = np.zeros(3).astype(float)
            t[b] = t[b] + eps
            grad_j.append((self.error_func(i, Xi, j, Xj + t) - self.error_func(i, Xi, j, Xj - t)) / (2 * eps))
        return np.column_stack(grad_i), np.column_stack(grad_j)

    def Gauss_Newton(self):

        self.draw_map_from_graph()
        print(self.pose_graph.edges)
        for itr in range(self.max_Gauss_Newton_iter):
            print(itr)
            b = np.zeros(len(self.pose_graph) * 3)
            H = np.zeros((len(self.pose_graph) * 3, len(self.pose_graph) * 3))
            for i in range(len(self.pose_graph) - 1):
                all_i_children = self.pose_graph[i, "children_id"]
                for j in all_i_children:
                    err = self.error_func(i, self.pose_graph[i, "pos"], j, self.pose_graph[j, "pos"])
                    Jij = self.Jacobian(i, j)
                    A, B = Jij
                    OM = np.ones((3, 3))
                    if j != i+1:
                        OM = np.ones((3, 3))*10000
                        print(err)

                    b[i * 3:(i + 1) * 3] += err.T @ OM @ A
                    b[j * 3:(j + 1) * 3] += err.T @ OM @ B
                    H[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] += A.T @ OM @ A
                    H[j * 3:(j + 1) * 3, j * 3:(j + 1) * 3] += B.T @ OM @ B
                    H[j * 3:(j + 1) * 3, i * 3:(i + 1) * 3] += B.T @ OM @ A
                    H[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] += A.T @ OM @ B

            H[0:3, 0:3] += np.ones((3, 3))
            b = b.reshape(b.shape[0], 1)
            #Q, R = np.linalg.qr(H)  # QR decomposition with qr function
            #y = np.dot(Q.T, b)  # Let y=Q'.B using matrix multiplication

            #for i in range(len(self.pose_graph) * 3):
            #    for j in range(len(self.pose_graph) * 3):
            #        print(H[i, j], end='; ')
            #    print()
            dx = np.linalg.lstsq(H.T, b, rcond=-1)[0]
            self.pose_graph.pos = (self.pose_graph.pos.reshape(1, len(self.pose_graph) * 3) - dx.T).reshape(
                len(self.pose_graph), 3)
        self.draw_map_from_graph()
        pyplot.axis('equal')
        pyplot.legend(numpoints=1)
        pyplot.show()

    def draw_map_from_graph(self):
        map = np.concatenate(split_objects([self.pose_graph[1, "mat_pos"], self.pose_graph[1, "lidar"]]), axis=0)
        for i in range(1, len(self.pose_graph)):
            new = np.concatenate(split_objects([self.pose_graph[i, "mat_pos"], self.pose_graph[i, "lidar"]]), axis=0)
            map = np.append(map, new, axis=0)
        pyplot.plot([p[0] for p in map], [p[1] for p in map], '.', label=f'new {2}')
        print(map.shape)


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


sim = Lidar_sim(700, "../lidar_odom_log/lidar_odom_log_8.txt")
sim.run()
