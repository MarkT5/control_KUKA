import numpy as np
import scipy
import scipy.spatial


class RRT:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.nodes = []
        self.graph = {}
        self.bool_map = None
        self.map_height = None
        self.map_width = None
        self.closest_node = None
        self.edges = []
        self.node_num = 0
        self.edge_num = 0
        self.stuck = 0
        self.force_random = 0
        self.dist_reached = False
        self.end_node = -1
        self.path = []

        # settings
        self.growth_factor = 100
        self.e = 5
        self.end_dist = 100

    def start(self):
        if not self.start_point.any():
            print("No start point")
        assert self.start_point.any()
        if not self.end_point.any():
            print("No end point")
        assert self.end_point.any()
        self.nodes.append(self.start_point)
        self.graph[0] = [self.start_point, None, None]
        self.node_num = 1
        map_shape = self.bool_map.shape
        self.map_height = map_shape[0]
        self.map_width = map_shape[1]

    def find_closest(self, point):
        nodes_tree = scipy.spatial.cKDTree(self.nodes)
        return nodes_tree.query(point)

    def check_obstacle(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        shift_vector = point2 - point1
        iters = sum(abs(shift_vector))
        shift_vector = shift_vector / iters
        all_shift = shift_vector
        c_point = np.array(False)
        for i in range(1, iters):
            all_shift = np.copy(shift_vector*i)
            c_point = np.around(point1+shift_vector*i).astype(np.int64)
            if self.bool_map[tuple(c_point)]:
                np.around(point1 + shift_vector*(i-1)).astype(np.int64)
                break
            if np.linalg.norm(shift_vector*i) >= self.growth_factor:
                break
        if np.linalg.norm(all_shift) < self.e or not c_point.any():
            return np.array(False), None
        return c_point, np.linalg.norm(all_shift)



    def step(self):
        node = self.graph[self.node_num - 1][0]
        dist = np.linalg.norm(node - self.end_point)
        if self.force_random:
            self.add_random()
            self.force_random -= 1
        if dist < 2:
            self.graph[self.node_num] = [self.end_point, self.node_num - 1, dist]
            self.dist_reached = True
            self.end_node = self.node_num
        elif dist < self.end_dist and self.force_random == 0:
            self.add_near_end()
            if self.stuck > 10:
                self.stuck = 0
                self.force_random = 50
        else:
            self.add_random()

    def add_random(self):
        random_point = np.random.rand(2)
        self.random_point = [int(random_point[0] * (self.map_height - 1)), int(random_point[1] * (self.map_width - 1))]
        self.closest_node = self.find_closest(self.random_point)
        node, dist = self.check_obstacle(self.nodes[self.closest_node[1]], self.random_point)
        if node.any():
            self.edges.append([self.node_num, self.closest_node[1]])
            self.nodes.append(node)
            self.edge_num += 1
            self.graph[self.node_num] = [node, self.closest_node[1], dist]
            self.node_num += 1

    def add_near_end(self):
        self.closest_node = self.find_closest(self.end_point)
        node, dist = self.check_obstacle(self.nodes[self.closest_node[1]], self.end_point)
        if node.any():
            self.edges.append([self.node_num, self.closest_node[1]])
            self.nodes.append(node)
            self.edge_num += 1
            self.graph[self.node_num] = [node, self.closest_node[1], dist]
            self.stuck = 0
            if node[0] == self.end_point[0] and node[1] == self.end_point[1]:
                print("done")
                self.dist_reached = True
                self.graph[self.node_num] = [self.end_point, self.node_num, dist]
                self.end_node = self.node_num
            self.node_num += 1
        else:
            self.stuck += 1

    def get_path(self):
        node_num = self.end_node
        self.path = []
        while node_num != 0:
            self.path.append(self.graph[node_num][0])
            node_num = self.graph[node_num][1]
        self.path.append(self.graph[node_num][0])

