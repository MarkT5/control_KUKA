import numpy as np
import math
import scipy


# pos, object, lidar, children[child id, edge id]]

class PoseGrah:
    def __init__(self):
        self.edges = np.array(False)
        self.edges_cov = np.array(False)
        self.edge_num = 0
        self.node_num = 0
        self.children_id = []
        self.edges_id = []
        self.objects = []
        self.lidars = []
        self.available_attributes = ["pos", "object", "lidar", "children", "children_id", "edge", "mat_pos",
                                     "lidar_mat_pos"]
        # self.node_weghts = np.array()

    def __getitem__(self, item):
        key = list(item)
        self.check_attribute(key)
        if len(key) == 1:
            if key[0] < 0:
                ind = int(key[0] + self.node_num)
            else:
                ind = int(key[0])
            return self.pos[ind], self.objects[ind], self.lidars[ind], self.children_id[ind]
        else:
            ind = int(key[0])
            if ind < 0:
                ind = int(self.node_num + ind)
            else:
                ind = int(ind)
            self.check_attribute(ind)
            if item[1] == "pos":
                return self.pos[ind]
            if item[1] == "object":
                return self.objects[ind]
            if item[1] == "lidar":
                return self.lidars[ind]
            if item[1] == "mat_pos":
                return self.homogen_matrix_from_pos(self.pos[ind])
            if item[1] == "lidar_mat_pos":
                return self.homogen_matrix_from_pos(self.pos[ind], True)
            if item[1] == "children":
                return self.children_id[ind], self.edges_id[ind]
            if item[1] == "children_id":
                return self.children_id[ind]
            if item[1] == "edge":
                child = key[2]
                if child < 0:
                    child = int(self.node_num + child)
                self.check_attribute(child)
                parent_child = sorted([ind, child])
                if len(item) != 3 or parent_child[1] not in self.children_id[parent_child[0]]:
                    raise AttributeError
                return self.edges[self.edges_id[parent_child[0]][self.children_id.index(parent_child[2])]]

    def __setitem__(self, key, value):
        key = list(key)
        self.check_attribute(key)
        if len(key) == 1:
            if key[0] < 0:
                ind = int(key[0] + self.node_num)
            else:
                ind = int(key[0])
            self.pos[ind], self.objects[ind], self.lidars[ind], self.children_id[ind] = value
        else:
            ind = int(key[0])
            if ind < 0:
                ind = int(self.node_num + ind)
            else:
                ind = int(key[0])
            self.check_attribute(ind)
            if key[1] == "pos":
                self.pos[ind] = value
            if key[1] == "object":
                self.objects[ind] = value
            if key[1] == "lidar":
                self.lidars[ind] = value
            if key[1] == "edge":  # [parent, "edge", child] = value
                child = key[2]
                if child < 0:
                    child = int(self.node_num + child)
                self.check_attribute(child)
                A = np.linalg.inv(self.homogen_matrix_from_pos(self.pos[ind]))
                if isinstance(value[0], list):
                    B = self.homogen_matrix_from_pos(value[0])
                else:
                    B = value[0]
                self.edges[self.edges_id[ind][self.children_id[ind].index(child)]] = np.dot(A, B)
                self.edges_cov[self.edges_id[ind][self.children_id[ind].index(child)]] = value[1]

    def __len__(self):
        return self.node_num

    def check_attribute(self, key):
        if isinstance(key, int):
            if key > self.node_num:
                raise AttributeError(f"node {key} does not exist")
        else:
            if key[0] > self.node_num:
                raise AttributeError(f"node {key[0]} does not exist")
            if key[1] not in self.available_attributes:
                raise AttributeError(f"no attribute {key[1]} in PoseGraph")

    def edge(self, i, j):
        if i < 0:
            i += self.node_num
        if j < 0:
            j += self.node_num
        parent_child = sorted([i, j])
        if parent_child[1] not in self.children_id[parent_child[0]]:
            raise AttributeError(f"no edge {i}, {j}")
        return self.edges[self.edges_id[parent_child[0]][self.children_id[parent_child[0]].index(parent_child[1])]]

    def edge_cov(self, i, j):
        if i < 0:
            i += self.node_num
        if j < 0:
            j += self.node_num
        parent_child = sorted([i, j])
        if parent_child[1] not in self.children_id[parent_child[0]]:
            raise AttributeError(f"no edge {i}, {j}")
        return self.edges_cov[self.edges_id[parent_child[0]][self.children_id[parent_child[0]].index(parent_child[1])]]

    def add_node(self, parent_id, pos, object, lidar, /, cov=np.array(False)):  # pos, object, lidar, children[child id, edge id]]
        self.objects.append(object)
        self.lidars.append(lidar)
        self.children_id.append([])
        self.edges_id.append([])

        if self.node_num == 0:
            self.pos = np.array([pos]).astype(np.double)
        else:
            parent_id = int(parent_id)
            if parent_id < 0:
                parent_id += self.node_num
            self.check_attribute(parent_id)
            if cov.any():
                self.add_edge(parent_id, pos, self.node_num, cov)
            else:
                self.add_edge(parent_id, pos, self.node_num)
            if isinstance(pos, np.ndarray):
                pos = self.pos_vector_from_homogen_matrix(pos)
            self.pos = np.append(self.pos, [pos], axis=0).astype(np.double)
        self.node_num += 1

    def add_edge(self, parent_id, to_n, child_id, cov=np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.8]])):
        parent_id = int(parent_id)
        child_id = int(child_id)
        if child_id < 0:
            child_id += self.node_num
        if parent_id < 0:
            parent_id += self.node_num
        self.check_attribute(parent_id)
        self.check_attribute(child_id)
        self.children_id[parent_id].append(child_id)
        self.edges_id[parent_id].append(self.edge_num)
        A = np.linalg.inv(self.homogen_matrix_from_pos(self.pos[parent_id])).astype(np.double)
        if isinstance(to_n, list):
            B = np.copy(self.homogen_matrix_from_pos(to_n)).astype(np.double)
        else:
            B = to_n.astype(np.double)
        if self.edge_num > 0:
            self.edges = np.append(self.edges, [np.dot(A, B)], axis=0).astype(np.double)
            self.edges_cov = np.append(self.edges, [cov], axis=0).astype(np.double)
        else:
            self.edges = np.array([np.dot(A, B)]).astype(np.double)
            self.edges_cov = np.array([cov]).astype(np.double)
        self.edge_num += 1

    def homogen_matrix_from_pos(self, pos, for_lidar=False):
        x, y, rot = pos
        cr = math.cos(rot)
        sr = math.sin(rot)
        if not for_lidar:
            return np.array([[cr, -sr, x],
                             [sr, cr, y],
                             [0, 0, 1]]).astype(np.double)
        robot_rad = 0.3
        X = np.array([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]]).astype(np.double)

        f = np.array([[1, 0, robot_rad],
                      [0, 1, 0],
                      [0, 0, 1]]).astype(np.double)

        rot = np.array([[cr, -sr, 0],
                        [sr, cr, 0],
                        [0, 0, 1]]).astype(np.double)

        X = np.dot(X, np.dot(rot, f))
        return X

    def undo_lidar(self, mat):
        cr = mat[0, 0]
        sr = mat[1, 0]
        robot_rad = 0.3
        f = np.array([[1, 0, robot_rad],
                      [0, 1, 0],
                      [0, 0, 1]]).astype(np.double)

        rot = np.array([[cr, -sr, 0],
                        [sr, cr, 0],
                        [0, 0, 1]]).astype(np.double)

        mat = np.dot(mat, np.linalg.inv(np.dot(rot, f)))
        mat[:2, :2] = [[cr, -sr], [sr, cr]]
        return mat

    def pos_vector_from_homogen_matrix(self, mat):
        aco = math.acos(max(-1, min(1, mat[0, 0])))
        asi = math.asin(max(-1, min(1, mat[1, 0])))
        asi_sign = -1 + 2 * (asi > 0)
        return np.array([*mat[:2, 2], aco * (asi_sign)]).astype(float)

    def find_n_closest(self, node_num, n):
        if node_num < 0:
            node_num += self.node_num
        if self.node_num < 2:
            return None
        n = min(n, self.node_num - 1)
        nodes_tree = scipy.spatial.cKDTree(self.pos)
        return nodes_tree.query(self.pos[node_num - 1], n)
