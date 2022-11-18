import numpy as np
import scipy
import scipy.spatial


class RRT:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.nodes = None
        self.node_map = None
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
        self.graph_printed = False

        # settings
        self.growth_factor = 100
        self.e = 5
        self.end_dist = 100
        self.rrt_star_rad = 20

    def start(self):
        if not self.start_point.any():
            print("No start point")
        assert self.start_point.any()
        if not self.end_point.any():
            print("No end point")
        assert self.end_point.any()
        self.nodes = np.array([self.start_point]).astype(np.uint32)
        self.node_map = np.zeros(self.bool_map.shape).astype(np.uint16)
        self.graph[0] = [None, [], 0]
        self.node_num = 1
        map_shape = self.bool_map.shape
        self.map_shape = (map_shape - np.ones(len(map_shape))).astype(np.uint32)
        print(self.map_shape)

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
        for i in range(1, iters + 1):
            all_shift = np.copy(shift_vector * i)
            c_point = np.around(point1 + shift_vector * i).astype(np.int64)
            if self.bool_map[tuple(c_point)]:
                np.around(point1 + shift_vector * (i - 1)).astype(np.int64)
                break
            if np.linalg.norm(shift_vector * i) >= self.growth_factor:
                break
        if np.linalg.norm(all_shift) < self.e or not c_point.any():
            return np.array(False), None
        return c_point, np.linalg.norm(all_shift)

    def step(self):
        node = self.nodes[self.node_num - 1]
        dist = np.linalg.norm(node - self.end_point)
        if self.force_random:
            self.add_random()
            self.force_random -= 1
        elif dist < self.end_dist and self.force_random == 0:
            self.add_near_end()
            if self.stuck > 10:
                self.stuck = 0
                self.force_random = 50
        else:
            self.add_random()

    def iter_node_map(self):
        pass
    def check_node_region(self, new_node):
        for i in range(new_node.shape):
            pass

    def add_node_to_closest(self, new_node):
        self.closest_node = self.find_closest(new_node)
        node, dist = self.check_obstacle(self.nodes[self.closest_node[1]], new_node)
        if node.any():
            self.edges.append([self.node_num, self.closest_node[1]])
            self.nodes = np.append(self.nodes, [node], axis=0).astype(np.uint32)
            self.edge_num += 1
            self.graph[self.node_num] = [self.closest_node[1], [], dist+self.graph[self.closest_node[1]][2]]
            self.node_map[tuple(node)] = self.node_num
            self.graph[self.closest_node[1]][1].append(self.node_num)
            self.node_num += 1
            return node
        else:
            return np.array(False)



    def add_random(self):
        random_point = (np.random.rand(2)*self.map_shape).astype(np.uint32)
        print(random_point)
        self.add_node_to_closest(random_point)

    def add_near_end(self):
        node = self.add_node_to_closest(self.end_point)
        if node.any() and node[0] == self.end_point[0] and node[1] == self.end_point[1]:
            print("done")
            self.dist_reached = True
            self.end_node = self.node_num-1
        else:
            self.stuck += 1

    def get_path(self):
        node_num = self.end_node
        self.path = []
        while node_num != 0:
            self.path.append(self.nodes[node_num])
            node_num = self.graph[node_num][0]
        self.path.append(self.nodes[node_num])
        if not self.graph_printed:
            print(self.graph)
            self.graph_printed = True
            print(self.node_map)
