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
            #self.clock.tick(10)

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
        x, y, r = self.odom
        xo, yo, ro = self.last_odom
        node_created = False

        if not object_coords:
            return
        if math.sqrt((x - xo) ** 2 + (y - yo) ** 2) > 0.1 or abs(r - ro) > 0.1 or len(self.pose_graph) == 0:
            # self.pose_graph.add_node(-1, self.odom, None, self.lidar)
            if len(self.pose_graph) < 2:
                self.pose_graph.add_node(-1, self.odom, None, self.lidar)
                return

            _, cl_points = self.pose_graph.find_n_closest(-1, 300)
            if isinstance(cl_points, int):
                cl_points = [cl_points]

            for obj in object_coords:
                self.draw_line_from_point_cloud(obj)
                peaks = douglas_peucker(obj, True)
                for nn in cl_points:
                    if len(peaks[0]) >= 3:
                        if not node_created:
                            self.pose_graph.add_node(-1, self.odom, obj[peaks[0][0]:peaks[0][-1]], self.lidar)
                            node_created = True
                        if isinstance(self.pose_graph[nn, "object"], np.ndarray):
                            weight_vector = self.generate_weight_vector(peaks)
                            icp_out = icp(obj[peaks[0][0]:peaks[0][-1]], np.array(self.pose_graph[nn, "object"]),
                                          weight_vector=[weight_vector])
                            # draw icp
                            if icp_out[-1] and icp_out[-1] < 0.02:
                                err = self.pose_graph.pos_vector_from_homogen_matrix(icp_out[0])
                                if err[0] ** 2 + err[1] ** 2 > 2 or abs(err[2]) > 0.8:
                                    return
                                print(err, icp_out[-1])
                                corrected_odom = self.pose_graph.undo_lidar(np.dot(icp_out[0], self.pose_graph[-1, "lidar_mat_pos"]).astype(np.double))
                                if nn == len(self.pose_graph) - 2:
                                    self.pose_graph[nn, "edge", -1] = (corrected_odom, np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]))  # [parent, "edge", child] = value
                                else:
                                    self.pose_graph.add_edge(nn, corrected_odom, -1, np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]))
                                self.pose_graph[-1, "object"] = obj[peaks[0][0]:peaks[0][-1]]
                                if False:
                                    pyplot.plot([p[0] for p in obj[peaks[0][0]:peaks[0][-1]]], [p[1] for p in obj[peaks[0][0]:peaks[0][-1]]], 'o',
                                              label='points 2')
                                    converted = np.concatenate(split_objects([np.dot(icp_out[0], self.pose_graph[-1, "lidar_mat_pos"]), self.lidar]))
                                    pyplot.plot([p[0] for p in converted], [p[1] for p in converted], 'o', label='converted')
                                    pyplot.plot([p[0] for p in self.pose_graph[nn, "object"]], [p[1] for p in self.pose_graph[nn, "object"]], '.', label='src')
                                    pyplot.axis('equal')
                                    pyplot.legend(numpoints=1)
                                    pyplot.show()
                                return
            if math.sqrt((x - xo) ** 2 + (y - yo) ** 2) > 0.2 or abs(r - ro) > 0.3 and not node_created:
                self.pose_graph.add_node(-1, self.odom, None, self.lidar)
            self.last_odom = self.odom
            print(len(self.pose_graph))

    def error_func(self, i, Xi, j, Xj):
        Zij = self.pose_graph.edge(i, j)
        Xi = self.pose_graph.homogen_matrix_from_pos(Xi)
        Xj = self.pose_graph.homogen_matrix_from_pos(Xj)
        err_mat = np.linalg.inv(Zij) @ (np.linalg.inv(Xi) @ Xj)
        err = self.pose_graph.pos_vector_from_homogen_matrix(err_mat)
        return err

    def Jacobian(self, i, j):
        eps = 1e-13
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
        for itr in range(self.max_Gauss_Newton_iter):
            ov_sq_err = np.array([0, 0, 0]).astype(float)
            print(itr)
            b = np.zeros(len(self.pose_graph) * 3)
            H = np.zeros((len(self.pose_graph) * 3, len(self.pose_graph) * 3))
            for i in range(len(self.pose_graph) - 1):
                all_i_children = self.pose_graph[i, "children_id"]
                for j in all_i_children:
                    err = self.error_func(i, self.pose_graph[i, "pos"], j, self.pose_graph[j, "pos"])
                    ov_sq_err += err * err
                    Jij = self.Jacobian(i, j)
                    A, B = Jij
                    # print(i, j, err)
                    OM = np.eye(3) * self.pose_graph.edge_cov(i,j)
                    #if self.pose_graph.edge_cov(i,j) > 0.4:
                    #    print(i, j, err)

                    # print("err.T @ A")
                    # print(err.T @ OM @ A)
                    # print("err.T @ B")
                    # print(err.T @ OM @ B)
                    # print("A.T @ A")
                    # print(A.T @ OM @ A)
                    # print("B.T @ B")
                    # print(B.T @ OM @ B)
                    # print("B.T @ A")
                    # print(B.T @ OM @ A)
                    # print("A.T@ B")
                    # print(A.T @ OM @ B)
                    # if j != i+1:
                    # OM[2, :] = [2, 2, 2]

                    b[i * 3:(i + 1) * 3] += err.T @ OM @ A
                    b[j * 3:(j + 1) * 3] += err.T @ OM @ B
                    H[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] += A.T @ OM @ A
                    H[j * 3:(j + 1) * 3, j * 3:(j + 1) * 3] += B.T @ OM @ B
                    H[j * 3:(j + 1) * 3, i * 3:(i + 1) * 3] += B.T @ OM @ A
                    H[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] += A.T @ OM @ B

            cv2.imshow("H", H)
            cv2.waitKey(3)
            # while cv2.waitKey(3) != 27:
            #    pass
            # H[0:3, 0:3] += np.ones((3, 3))
            b = b.reshape(b.shape[0], 1)
            print(ov_sq_err)
            # Q, R = np.linalg.qr(H)  # QR decomposition with qr function
            # y = np.dot(Q.T, b)  # Let y=Q'.B using matrix multiplication

            # for i in range(len(self.pose_graph) * 3):
            #    for j in range(len(self.pose_graph) * 3):
            #        print(H[i, j], end='; ')
            #    print()
            H = np.copy(H[3:, 3:])
            b = np.copy(b[3:, 0])
            dx = np.linalg.lstsq(H, -b, rcond=-1)[0]
            # print("H diag")
            # print(np.linalg.det(H))
            # print(list([H[i][i] for i in range(H.shape[0])]))
            # print()
            # print(b)
            dx = np.append(np.array([[0], [0], [0]]), dx)
            self.pose_graph.pos = np.copy((self.pose_graph.pos.reshape(1, len(self.pose_graph) * 3) + dx.T).reshape(
                len(self.pose_graph), 3))
        self.draw_map_from_graph()
        pyplot.axis('equal')
        pyplot.legend(numpoints=1)
        pyplot.show()

    def draw_map_from_graph(self):
        map = np.concatenate(split_objects([self.pose_graph[1, "lidar_mat_pos"], self.pose_graph[1, "lidar"]]), axis=0)
        for i in range(1, len(self.pose_graph)):
            new = np.concatenate(split_objects([self.pose_graph[i, "lidar_mat_pos"], self.pose_graph[i, "lidar"]]), axis=0)
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

    def update_lidar(self):
        """
        draws lidar data on body_pos_screen
        :return:
        """
        odom = self.odom
        lidar = self.lidar
        self.draw_lidar(odom, lidar)

    def draw_lidar(self, odom, lidar):
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


sim = Lidar_sim(700, "../lidar_odom_log/lidar_odom_log_6.txt")

sim.run()
