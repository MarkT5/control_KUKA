import numpy as np

from import_libs import *
# pos, object, children[child id, edge id]]

class PoseGrah:
    def __init__(self):
        self.edges = np.array(False)
        self.edges_cov = np.array(False)
        self.pos = np.array(False)
        self.edge_num = 0
        self.node_num = 0
        self.children_id = []
        self.edges_id = []
        self.objects = []
        self.path = []
        self.available_attributes = ["pos", "object", "lidar", "children", "children_id", "edge", "mat_pos",
                                     "lidar_mat_pos", "path"]

    def __getitem__(self, item):
        key = list(item)
        self.check_attribute(key)
        if len(key) == 1:
            if key[0] < 0:
                ind = int(key[0] + self.node_num)
            else:
                ind = int(key[0])
            return self.pos[ind], self.objects[ind], self.objects[ind].lidar, self.children_id[ind]
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
                return self.objects[ind].lidar
            if item[1] == "mat_pos":
                return homogen_matrix_from_pos(self.pos[ind])
            if item[1] == "lidar_mat_pos":
                return homogen_matrix_from_pos(self.pos[ind], True)
            if item[1] == "children":
                return self.children_id[ind], self.edges_id[ind]
            if item[1] == "children_id":
                return self.children_id[ind]
            if item[1] == "path":
                return self.path[ind]
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
            self.pos[ind], self.objects[ind], self.objects[ind].lidar, self.children_id[ind] = value
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
            if key[1] == "path":
                self.path[ind] = value
            if key[1] == "lidar":
                self.objects[ind].lidar = value
            if key[1] == "edge":  # [parent, "edge", child] = value
                child = key[2]
                if child < 0:
                    child = int(self.node_num + child)
                self.check_attribute(child)
                A = np.linalg.inv(homogen_matrix_from_pos(self.pos[ind]))
                if isinstance(value[0], list):
                    B = homogen_matrix_from_pos(value[0])
                else:
                    B = value[0]
                self.edges[self.edges_id[ind][self.children_id[ind].index(child)]] = np.dot(A, B)
                self.edges_cov[self.edges_id[ind][self.children_id[ind].index(child)]] = value[1]

    def __len__(self):
        return self.node_num

    def update(self, ind):
        self.objects[ind].update(self.pos[ind])

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

    def add_node(self, parent_id, pos, object, path, /, cov=np.array(False)):  # pos, object, children[child id, edge id]]
        self.objects.append(object)
        self.children_id.append([])
        self.edges_id.append([])
        self.path.append(path)
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos)
        if self.node_num == 0:
            self.pos = np.array([pos])
        else:
            parent_id = int(parent_id)
            if parent_id < 0:
                parent_id += self.node_num
            self.check_attribute(parent_id)
            if cov.any():
                self.add_edge(parent_id, pos, self.node_num, cov)
            else:
                self.add_edge(parent_id, pos, self.node_num)
            if pos.shape == (3, 3):
                pos = pos_vector_from_homogen_matrix(pos)
            self.pos = np.append(self.pos, [pos], axis=0)
        self.node_num += 1

    def add_edge(self, parent_id, to_n, child_id, cov=np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])):
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
        A = np.linalg.inv(homogen_matrix_from_pos(self.pos[parent_id]))
        if not isinstance(to_n, np.ndarray):
            to_n = np.array(to_n)
        if to_n.shape == (3, 3):
            B = to_n
        else:
            B = homogen_matrix_from_pos(to_n)
        if self.edge_num > 0:
            self.edges = np.append(self.edges, [np.dot(A, B)], axis=0)
            self.edges_cov = np.append(self.edges_cov, [cov], axis=0)
        else:
            self.edges = np.array([np.dot(A, B)])
            self.edges_cov = np.array([cov])
        self.edge_num += 1



    def find_n_closest(self, node_num, n):
        if node_num < 0:
            node_num += self.node_num
        if self.node_num < 2:
            return None
        n = min(n, self.node_num - 1)
        nodes_tree = scipy.spatial.cKDTree(self.pos)
        return nodes_tree.query(self.pos[node_num - 1], n)
