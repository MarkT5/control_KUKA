import math
import numpy as np
import scipy
from matplotlib import pyplot


class PointCloud:
    def __init__(self, data):
        self.odom, self.lidar = data
        self.xy_form = []
        self.gaps = [0]
        self.object_sizes = [[0, 0]]
        self.split_objects()
        self.peaks = []
        self.peak_coords = [None, np.array([False])]
        for i in range(len(self.gaps) - 1, 0, -1):
            self.peaks.append([*self.douglas_peucker(self.gaps[i - 1], self.gaps[i])])
            for i in range(1, len(self.peaks[-1][0]) - 1):
                if self.peak_coords[1].any():
                    self.peak_coords[0].append(len(self.peaks) - 1)
                    self.peak_coords[1] = np.append(self.peak_coords[1], [self.xy_form[self.peaks[-1][0][i]]], axis=0)
                else:
                    self.peak_coords = [[len(self.peaks) - 1], np.array([self.xy_form[self.peaks[-1][0][i]]])]

        # print(self.odom, self.lidar)
        # print("gaps:", self.gaps)
        # print("sizes:", self.object_sizes)
        # print("peaks:", self.peaks)
        # print("peak coords:", self.peak_coords)

    def __eq__(self, other):  ##icp
        if self.peak_coords[1].any() and other.peak_coords[1].any():
            neighbours = np.array([*self.nearest_neighbor([self.peak_coords[1]], other.peak_coords[1]), range(len(self.peak_coords[1]))]).T
            neighbours = sorted(neighbours, key=lambda x: x[0])
            if neighbours[0][0] < 0.5:
                peak = self.peaks[self.peak_coords[0][int(neighbours[0][2])]]
                weight_vector = self.generate_weight_vector(peak)
                icp_out = self.icp(self.get_obj(self.peak_coords[0][int(neighbours[0][2])]),
                                   other.get_obj(other.peak_coords[0][int(neighbours[0][1])]),
                                   weight_vector=[weight_vector])
                return icp_out
        else:
            return None


    def get_obj(self, ind):
        ind = int(ind)
        peak = self.peaks[ind]

        return np.array(self.xy_form[peak[0][0]:peak[0][-1]])

    def conv_cil_to_dec(self, ind=None, /, odom = np.array(False)):
        out_points = []
        if not odom.any():
            odom = np.array(self.odom)
        lidar = self.lidar
        x, y, rot = odom
        robot_rad = 0.3
        if odom.shape != (3, 3):
            X = np.array([[1, 0, x],
                          [0, 1, y],
                          [0, 0, 1]])

            f = np.array([[1, 0, robot_rad],
                          [0, 1, 0],
                          [0, 0, 1]])

            rot = np.array([[math.cos(rot), -math.sin(rot), 0],
                            [math.sin(rot), math.cos(rot), 0],
                            [0, 0, 1]])

            odom = np.dot(X, np.dot(rot, f))
        if isinstance(ind, int):
            lid_ang = ind * math.radians(240) / len(lidar) - math.radians(30)
            lid_dist = lidar[ind]
            if lid_dist > 5.5:
                return None, None
            ox = lid_dist * math.sin(lid_ang)
            oy = lid_dist * math.cos(lid_ang)
            return (np.dot(odom, np.array([[ox, oy, 1]]).T).T)[0, :2]
        elif isinstance(ind, list):
            for i in ind:
                lid_ang = i * math.radians(240) / len(lidar) - math.radians(30)
                lid_dist = lidar[i]
                if lid_dist > 5.5:
                    continue
                ox = lid_dist * math.sin(lid_ang)
                oy = lid_dist * math.cos(lid_ang)
                out_points.append((ox, oy))
        else:
            for i in range(len(lidar)):
                lid_ang = i * math.radians(240) / len(lidar) - math.radians(30)
                lid_dist = lidar[i]
                if lid_dist > 5.5:
                    continue
                ox = lid_dist * math.sin(lid_ang)
                oy = lid_dist * math.cos(lid_ang)
                out_points.append((ox, oy))
        out_points = np.array(out_points)
        converted = np.ones((out_points.shape[1] + 1, out_points.shape[0]))
        converted[:out_points.shape[1], :] = np.copy(out_points.T)
        # transform
        converted = np.dot(odom, converted)
        # back from homogeneous to cartesian
        converted = np.array(converted[:converted.shape[1], :]).T
        return converted[:, :2]

    def split_objects(self, odom=np.array(False)):
        correct_self = False
        if isinstance(odom, np.ndarray) and not odom.any():
            odom = self.odom
            correct_self = True
        lidar = self.lidar
        object_sizes = [[0, 0]]
        gaps = [0]
        obj_ind = 1
        xy_form = []
        points_recorded = 0
        for i in range(1, len(lidar)):
            if lidar[i] > 5.5 or lidar[i - 1] > 5.5 or lidar[i] < 0.5 or lidar[i - 1] < 0.5:
                continue
            if abs(lidar[i - 1] - lidar[i]) < 0.1:
                object_sizes[-1][1] += 1
                xy_form.append(self.conv_cil_to_dec(i, odom=np.array(odom)))
                points_recorded += 1
            else:
                if points_recorded != gaps[-1]:
                    gaps.append(points_recorded)
                object_sizes.append([obj_ind, 0])
                obj_ind += 1
        if points_recorded != gaps[-1]:
            gaps.append(points_recorded - 1)
        if correct_self:
            self.gaps = gaps
            self.xy_form = np.array(xy_form)
            self.object_sizes = sorted(object_sizes, key=lambda x: x[1], reverse=True)
        return np.array(xy_form)

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

    def find_farthest(self, points, threshold=1):
        x1, y1 = points[0]
        x2, y2 = points[-1]
        main_vert = False
        perp_vert = False
        a_main = 0
        a_perp = 0
        max_dist = -1
        max_i = None
        if x1 == x2:
            main_vert = True
        else:
            a_main = (y1 - y2) / (x1 - x2)
        b_main = y1 - a_main * x1
        if a_main != 0:
            a_perp = -1 / a_main
        else:
            perp_vert = True
        for i in range(1, len(points) - 2):
            xp, yp = points[i]
            if main_vert:
                dist = xp ** 2
            elif perp_vert:
                dist = yp ** 2
            else:
                b_perp = yp - a_perp * xp
                xc = (b_main - b_perp) / (a_perp - a_main)
                yc = xc * a_main + b_main
                dist = (xp - xc) ** 2 + (yp - yc) ** 2
            if dist > max_dist:
                max_dist = dist
                max_i = i
        if max_dist < threshold:
            return None
        if max_dist == -1:
            return None
        return max_i

    def binary_search(self, arr, x):
        low, high = 0, len(arr) - 1
        mid = 0
        mid_old = 0
        while high >= low:
            mid = (high + low) // 2
            if arr[mid] == x:
                return None
            elif arr[mid] > x:
                high = mid
            else:
                low = mid
            if mid_old == high + low:
                break
            mid_old = high + low
        return mid + 1

    def douglas_peucker(self, point1, point2, only_peaks=False):
        iters = 4
        min_weight = iters * (point2 - point1) // 15
        point_list = [point1, point2 - 1]
        weights = [0]
        new_point_list = [point1, point2 - 1]

        for a in range(iters):
            for i in range(1, len(point_list)):
                add_point = None
                if point_list[i] - point_list[i - 1] > 5:
                    add_point = self.find_farthest(list(self.xy_form[point_list[i - 1]:point_list[i]]), 0.1)
                if not add_point:
                    weights[i - 1] += point_list[i] - point_list[i - 1]
                    continue
                add_point += point_list[i - 1]
                adress = self.binary_search(new_point_list, add_point)
                if not adress:
                    continue
                weights.insert(adress, 0)
                new_point_list.insert(adress, add_point)
            point_list = new_point_list[:]
        out_points = []
        f = 0
        while f < len(weights) - 1 and weights[f] < min_weight:
            f += 1
        out_points.append(point_list[f])
        for i in range(len(weights)):
            if weights[i] > min_weight:
                if not out_points:
                    out_points.append(point_list[i])
                out_points.append(point_list[i + 1])
        return out_points, weights

    def find_corners(self, object, approx_points_ind):
        connection_coords = [[object[p][0], object[p][1]] for p in approx_points_ind]
        if not connection_coords:
            return None, None
        b_prev = None
        b_curr = 0
        a_prev = None
        ang_curr = None
        ang_prev = None
        offset = 0.4
        corner = []
        corner_lines = []
        for i in range(1, len(connection_coords)):
            x1, y1 = connection_coords[i - 1]
            x2, y2 = connection_coords[i]
            a_curr = 0
            if x1 == x2 or y1 == y2:
                pass
            else:
                a_curr = (y1 - y2) / (x1 - x2)
                b_curr = y1 - (y1 - y2) / (x1 - x2) * x1
            if a_prev:
                pass
            ang_curr = math.atan2((y1 - y2), (x1 - x2))
            if a_prev and abs(math.atan(a_curr) + math.atan(1 / a_prev)) < offset:
                corner.append([object[approx_points_ind[i - 2]:approx_points_ind[i]],
                               approx_points_ind[i - 1] - approx_points_ind[i - 2]])
                corner_lines.append(([ang_prev, b_prev], [ang_curr, b_curr], object[approx_points_ind[i - 1]],
                                     connection_coords[i - 2:i + 1]))
            a_prev = a_curr
            b_prev = b_curr
            ang_prev = ang_curr
        if corner:
            return corner, corner_lines
        else:
            return None, None

    def best_fit_transform(self, A, B, weight_vector=None):
        # assert A.shape == B.shape

        # get number of dimensions
        n, m = A.shape

        # translate points to their centroids
        # centroid_A = np.mean(A, axis=0)
        # centroid_B = np.mean(B, axis=0)

        weights_sum = np.sum(weight_vector)
        centroid_A = np.sum(np.multiply(A, weight_vector.T), axis=0) / weights_sum
        centroid_B = np.sum(np.multiply(B, weight_vector.T), axis=0) / weights_sum
        AA = A - centroid_A
        BB = B - centroid_B
        # print(BB)

        # rotation matrix
        H = np.dot(np.multiply(AA, weight_vector.T).T, np.multiply(BB, weight_vector.T))
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(self, src, dst):
        # assert src.shape == dst.shape
        A_tree = scipy.spatial.KDTree(dst)
        dist, indexes = A_tree.query(src)
        return dist.ravel(), indexes.ravel()

    def icp(self, A, B, init_pose=None, max_iterations=20, tolerance=0.0001, /, weight_vector=None):
        # print(A.shape, B.shape)
        # assert A.shape == B.shape
        # get number of dimensions
        n, m = A.shape

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, A.shape[0]))
        dst = np.ones((m + 1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0
        distances = []
        mean_error = None
        i = 0
        error = None
        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            if weight_vector:
                T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T, np.array(weight_vector))
            else:
                T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T, np.ones(n).reshape(1, n))
            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                tolerance_error = np.abs(prev_error - mean_error)
                break

            prev_error = mean_error
        T, _, _ = self.best_fit_transform(A, src[:m, :].T, np.ones(n).reshape(1, n))
        return T, distances, i, mean_error
