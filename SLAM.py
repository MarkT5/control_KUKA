import numpy as np

from import_libs import *


class SLAM:
    def __init__(self, robot):
        self.robot = robot
        self.last_odom = [0, 0, 0]
        self.last_path = 0
        self.pose_graph = PoseGrah()  # odom, object, lidar, children[child id, edge id]]
        self.max_Gauss_Newton_iter = 5
        self.all_detected_corners = [None, np.array(False)]
        self.stored_detected_corners = [None, np.array(False)]
        self.path_length = 0
        self.refactor = False
        self.stored_data = (np.array(False), np.array(False))
        self.main_thr = thr.main_thread()
        self.optimizing = False

        self.map_size = 2000
        self.odom, self.lidar = self.robot.lidar
        self.odom_err = None
        self.last_reg_odom = None
        self.corr_pos = [0, 0, 0]
        self.point_cloud = None
        self.bool_map = np.ones((self.map_size, self.map_size))
        self.discrete = 100
        self.node_queue = []

        self.queue_lock = thr.Lock()

    def run(self):
        time.sleep(1)
        self.odom, self.lidar = self.robot.increment_by_wheels, self.robot.lidar[-1]
        self.pose_graph.add_node(-1, self.odom, PointCloud([self.odom, self.lidar]), self.path_length)
        optimizer = thr.Thread(target=self.optimize, args=())
        optimizer.start()
        while self.main_thr.is_alive():
            self.odom, self.lidar = self.robot.increment_by_wheels, self.robot.lidar[-1]
            if self.odom and self.lidar:
                self.process_SLAM()
                time.sleep(0.005)


    def map(self):
        return self.bool_map

    def optimize(self):
        j=0
        while self.main_thr.is_alive():
            #print(j)
            for i in self.node_queue:
                self.pose_graph.add_node(-1, *i)
            self.last_reg_odom = self.pose_graph[-1, "mat_pos"]
            if len(self.pose_graph) > 0:
                self.node_queue = []
                self.optimizing = True
                self.SLAM_add_edges()
                self.Gauss_Newton()
                corr_odom = np.linalg.inv(self.last_reg_odom) @ homogen_matrix_from_pos(self.robot.increment_by_wheels)
                self.robot.calculated_pos = [*pos_vector_from_homogen_matrix(self.pose_graph[-1, "mat_pos"] @ corr_odom)]
                self.optimizing = False
                self.all_detected_corners = self.stored_detected_corners[:]
            j+=1



    def add_curr_to_queue(self):
        self.queue_lock.acquire()
        self.node_queue.append([self.odom, self.point_cloud, self.path_length])
        self.queue_lock.release()
        self.last_path = self.path_length
    def process_SLAM(self):
        self.point_cloud = PointCloud([self.odom, self.lidar])
        x, y, r = self.odom
        xo, yo, ro = self.last_odom
        dist = math.sqrt((x - xo) ** 2 + (y - yo) ** 2 + (r - ro) ** 2)
        self.last_odom = self.odom
        self.path_length += dist
        if self.point_cloud.peak_coords[0]:
            #nn = self.check_existing_corners(self.point_cloud)
            #if not nn:
            #    self.add_curr_to_queue()
            #    print("nn")
            #    return
            #else:
            #    nn = int(nn)
            for i in range(len(self.pose_graph) - 1, len(self.pose_graph)-5, -1):
                icp_out = self.pose_graph[-1, "object"].icp(self.pose_graph[i, "object"])
                err = pos_vector_from_homogen_matrix(icp_out[0])
                if icp_out and icp_out[-1] < 0.03 and err[0] ** 2 + err[1] ** 2 < 1 and abs(err[2]) < 0.7:
                    if self.path_length - self.last_path > 0.7:
                        self.add_curr_to_queue()
                        print("no neighbours")
                    return
        if self.path_length - self.last_path > 1:
            self.add_curr_to_queue()
            print("too long")

    def SLAM_add_edges(self):
        self.stored_detected_corners = [None, np.array(False)]
        self.pose_graph.update(0)
        for i in range(len(self.pose_graph)):
            self.pose_graph.update(i)
        print("now cheking: ")
        for i in range(len(self.pose_graph)-1, -1, -1):

            for j in range(0, i):
                print(i, j)
                icp_out = self.pose_graph[i, "object"].icp(self.pose_graph[j, "object"])
                corrected_odom = undo_lidar(
                    np.dot(icp_out[0], self.pose_graph[i, "lidar_mat_pos"]))
                if icp_out and icp_out[-1] < 0.04:
                    self.pose_graph.add_edge(j, corrected_odom, i, np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]))
                    #if abs(i-j)>10:
                    #    self.plot_icp(corrected_odom, j, i)
                    break

        '''if point_cloud.peak_coords[0]:
                nn = self.check_existing_corners(point_cloud, optimizing=True)
                if not nn:
                    self.check_existing_corners(point_cloud, node_ind=i, add=True, optimizing=True)
                    continue
                else:
                    nn = int(nn)
                self.pose_graph.update(nn)
                icp_out = point_cloud.icp(self.pose_graph[nn, "object"])
                err = pos_vector_from_homogen_matrix(icp_out[0])
                corrected_odom = undo_lidar(
                    np.dot(icp_out[0], self.pose_graph[i, "lidar_mat_pos"]))

                if icp_out and icp_out[-1] < 0.03 and err[0] ** 2 + err[1] ** 2 < 1 and abs(err[2]) < 0.7:
                    if nn < i and not i in self.pose_graph[nn, "children_id"]:
                        self.pose_graph.add_edge(nn, corrected_odom, i,
                                                 np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]))
                else:
                    self.check_existing_corners(point_cloud, node_ind=i, add=True, optimizing=True)'''

    def plot_icp(self, corrected_odom, j, i):
        corrected_odom = homogen_matrix_from_pos(pos_vector_from_homogen_matrix(corrected_odom), True)
        out_points = np.array(self.pose_graph[i, "object"].xy_local_form)
        converted = np.ones((out_points.shape[1] + 1, out_points.shape[0]))
        converted[:out_points.shape[1], :] = np.copy(out_points.T)
        # transform
        converted = np.dot(corrected_odom, converted)
        # back from homogeneous to cartesian
        converted = (np.array(converted[:converted.shape[1], :]).T)[:, :2]
        pyplot.plot([p[0] for p in self.pose_graph[j, "object"].xy_form],
                    [p[1] for p in self.pose_graph[j, "object"].xy_form], 'o',
                    label='points 2')
        pyplot.plot([p[0] for p in self.pose_graph[i, "object"].xy_form],
                    [p[1] for p in self.pose_graph[i, "object"].xy_form], '.', label='src')
        pyplot.plot([p[0] for p in converted],
                    [p[1] for p in converted], '.', label='converted')

        pyplot.axis('equal')
        pyplot.legend(numpoints=1)
        pyplot.show()

    def check_existing_corners(self, object, /, node_ind=None, add=False, optimizing=False):
        if optimizing:
            corners = self.stored_detected_corners
        else:
            corners = self.all_detected_corners
        if corners[1].any():
            nodes_tree = scipy.spatial.cKDTree(corners[1])
            res = np.array(nodes_tree.query(object.peak_coords[1])).T
            res = sorted(res, key=lambda x: x[0])
            if add:
                peak_coords = object.peak_coords[1]
                ind = [node_ind] * len(peak_coords)
                if optimizing:
                    for i in ind:
                        self.stored_detected_corners[0].append(i)
                    self.stored_detected_corners[1] = np.append(self.stored_detected_corners[1], peak_coords, axis=0)
                else:
                    for i in ind:
                        self.all_detected_corners[0].append(i)
                    self.all_detected_corners[1] = np.append(self.all_detected_corners[1], peak_coords, axis=0)
            return corners[0][int(res[0][1])]
        elif node_ind:
            peak_coords = object.peak_coords[1]
            ind = [node_ind] * len(peak_coords)
            if optimizing:
                self.stored_detected_corners = [ind, peak_coords]
            else:
                self.all_detected_corners = [ind, peak_coords]
        return None

    def error_func(self, i, Xi, j, Xj):
        Zij = self.pose_graph.edge(i, j)
        Xi = homogen_matrix_from_pos(Xi)
        Xj = homogen_matrix_from_pos(Xj)
        err_mat = np.linalg.inv(Zij) @ (np.linalg.inv(Xi) @ Xj)
        err = pos_vector_from_homogen_matrix(err_mat)
        return err

    def Jacobian(self, i, j):
        eps = 1e-11
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

    def Gauss_Newton(self, max_itr=None):

        if not max_itr:
            max_itr = self.max_Gauss_Newton_iter
        for itr in range(max_itr):
            ov_sq_err = np.array([0, 0, 0]).astype(float)
            # print(itr)
            b = np.zeros(len(self.pose_graph) * 3)
            H = np.zeros((len(self.pose_graph) * 3, len(self.pose_graph) * 3))
            for i in range(len(self.pose_graph) - 1):
                all_i_children = self.pose_graph[i, "children_id"]
                for j in all_i_children:
                    err = self.error_func(i, self.pose_graph[i, "pos"], j, self.pose_graph[j, "pos"])
                    ov_sq_err += err * err
                    Jij = self.Jacobian(i, j)
                    A, B = Jij
                    OM = self.pose_graph.edge_cov(i, j)
                    b[i * 3:(i + 1) * 3] += err.T @ OM @ A
                    b[j * 3:(j + 1) * 3] += err.T @ OM @ B
                    H[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] += A.T @ OM @ A
                    H[j * 3:(j + 1) * 3, j * 3:(j + 1) * 3] += B.T @ OM @ B
                    H[j * 3:(j + 1) * 3, i * 3:(i + 1) * 3] += B.T @ OM @ A
                    H[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] += A.T @ OM @ B

            b = b.reshape(b.shape[0], 1)
            cv2.imshow("H", H)
            cv2.waitKey(3)
            # print(ov_sq_err)
            H = np.copy(H[3:, 3:])
            b = np.copy(b[3:, 0])
            dx = np.linalg.lstsq(H, -b, rcond=-1)[0]
            dx = np.append(np.array([[0], [0], [0]]), dx)
            self.pose_graph.pos = np.copy(
                (self.pose_graph.pos.reshape(1, len(self.pose_graph) * 3) + dx.T * 0.4).reshape(len(self.pose_graph),
                                                                                                3))

        bool_map = np.ones((self.map_size, self.map_size))
        for i in range(0, len(self.pose_graph)):
            half = self.map_size // 2
            new = self.pose_graph[i, "object"].xy_form
            for j in new:
                try:
                    j = j * self.discrete
                    bool_map[int(j[0]) + half][int(j[1]) + half] = 0
                except IndexError:
                    print("out of bounds")

        # map = self.pose_graph.get_converted_object(1)
        # for i in range(2, len(self.pose_graph)):
        #    new = self.pose_graph.get_converted_object(i)
        #    map = np.append(map, new, axis=0)
        # pyplot.plot([p[0] for p in map], [p[1] for p in map], '.', label="test")
        # print(map.shape)

        self.bool_map = bool_map
