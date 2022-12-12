import numpy as np
import math
import scipy


# pos, object, lidar, children[child id, edge id]]

class PoseGrah:
    def __init__(self):
        self.edges = np.array(False)
        self.edge_num = 0
        self.node_num = 0
        self.children_id = []
        self.edges_id = []
        self.objects = []
        self.lidars = []
        self.available_attributes = ["pos", "object", "lidar", "children", "children_id", "edge", "mat_pos"]
        # self.node_weghts = np.array()

    def __getitem__(self, item):
        key = list(item)
        self.check_attribute(key)
        if len(key) == 1:
            if key[0] < 0:
                parent = int(key[0] + self.node_num)
            else:
                parent = int(key[0])
            return self.pos[parent], self.objects[parent], self.lidars[parent], self.children_id[parent]
        else:
            parent = int(key[0])
            if parent < 0:
                parent = int(self.node_num + parent)
            else:
                parent = int(parent)
            self.check_attribute(parent)
            if item[1] == "pos":
                return self.pos[parent]
            if item[1] == "object":
                return self.objects[parent]
            if item[1] == "lidar":
                return self.lidars[parent]
            if item[1] == "mat_pos":
                return self.homogen_matrix_from_pos(self.pos[parent])
            if item[1] == "children":
                return self.children_id[parent], self.edges_id[parent]
            if item[1] == "children_id":
                return self.children_id[parent]
            if item[1] == "edge":
                child = key[2]
                if child < 0:
                    child = int(self.node_num + child)
                self.check_attribute(child)
                parent_child = sorted([parent, child])
                if len(item) != 3 or parent_child[1] not in self.children_id[parent_child[0]]:
                    raise AttributeError
                return self.edges[self.edges_id[parent_child[0]][self.children_id.index(parent_child[2])]]

    def __setitem__(self, key, value):
        key = list(key)
        self.check_attribute(key)
        if len(key) == 1:
            if key[0] < 0:
                parent = int(key[0] + self.node_num)
            else:
                parent = int(key[0])
            self.pos[parent], self.objects[parent], self.lidars[parent], self.children_id[parent] = value
        else:
            parent = int(key[0])
            if parent < 0:
                parent = int(self.node_num + parent)
            else:
                parent = int(key[0])
            self.check_attribute(parent)
            if key[1] == "pos":
                self.pos[parent] = value
            if key[1] == "object":
                self.objects[parent] = value
            if key[1] == "lidar":
                self.lidars[parent] = value
            if key[1] == "edge":  # [parent, "edge", child] = value
                child = key[2]
                if child < 0:
                    child = int(self.node_num + child)
                self.check_attribute(child)
                self.add_edge(parent, value, child)

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

    def edge(self, i, j, /, pos_form=False):
        parent_child = sorted([i, j])
        if parent_child[1] not in self.children_id[parent_child[0]]:
            raise AttributeError(f"no edge {i}, {j}")
        if pos_form:
            return self.pos_vector_from_homogen_matrix(
                self.edges[self.edges_id[parent_child[0]][self.children_id[parent_child[0]].index(parent_child[1])]])
        else:
            return self.edges[self.edges_id[parent_child[0]][self.children_id[parent_child[0]].index(parent_child[1])]]

    def add_node(self, parent_id, pos, object, lidar):  # pos, object, lidar, children[child id, edge id]]
        self.objects.append(object)
        self.lidars.append(lidar)
        self.children_id.append([])
        self.edges_id.append([])

        if self.node_num == 0:
            self.pos = np.array([pos])
        else:
            if parent_id < 0:
                parent_id += self.node_num
            self.check_attribute(parent_id)
            self.pos = np.append(self.pos, [pos], axis=0)
            self.add_edge(parent_id, pos, self.node_num)
        print(self.node_num, self.edge_num)

        self.node_num += 1

    def add_edge(self, parent_id, to_n, child_id):
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
        A = np.linalg.inv(self.homogen_matrix_from_pos(self.pos[parent_id]))
        if isinstance(to_n, list):
            B = np.copy(self.homogen_matrix_from_pos(to_n))
        else:
            B = to_n
        if self.edge_num > 0:
            self.edges = np.append(self.edges, [np.dot(A, B)], axis=0)
        else:
            self.edges = np.array([np.dot(A, B)])
        self.edge_num += 1

    def homogen_matrix_from_pos(self, pos):
        x, y, rot = pos
        robot_rad = 0.3
        X = np.array([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]])

        f = np.array([[1, 0, robot_rad],
                      [0, 1, 0],
                      [0, 0, 1]])

        rot = np.array([[math.cos(rot), -math.sin(rot), 0],
                        [math.sin(rot), math.cos(rot), 0],
                        [0, 0, 1]])

        X = np.dot(X, np.dot(rot, f))
        return X

    def pos_vector_from_homogen_matrix(self, mat):
        aco = math.acos(max(-1, min(1, mat[0, 0])))
        asi = math.asin(max(-1, min(1, mat[1, 0])))
        asi_sign = 1 - 2 * (asi > 0)
        return np.array([*mat[:2, 2], aco * (asi_sign)]).astype(float)

    def find_n_closest(self, node_num, n):
        if node_num < 0:
            node_num += self.node_num
        if self.node_num < 2:
            return None
        n = min(n, self.node_num - 1)
        nodes_tree = scipy.spatial.cKDTree(self.pos)
        return nodes_tree.query(self.pos[node_num - 1], n)
