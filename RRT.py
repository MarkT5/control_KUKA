import numpy as np
import scipy
import scipy.spatial
import cv2



class RRT:
    def __init__(self, /,
                 start_point=None,
                 end_point=None,
                 bin_map=None,
                 growth_factor=15,
                 e=0.04,
                 end_dist=10,
                 rrt_star_rad=20):

        self.start_point = start_point
        self.end_point = end_point
        self.bool_map = bin_map

        self.star = True

        self.nodes = np.array([self.start_point]).astype(np.uint32)
        self.node_map = np.ones(self.bool_map.shape).astype(np.int16)*-1
        self.node_map[tuple(self.start_point)] = 0
        self.graph = {}
        self.graph[0] = [None, [], 0]
        self.node_num = 1
        map_shape = self.bool_map.shape
        self.map_shape = (map_shape - np.ones(len(map_shape))).astype(np.uint32)
        self.stuck = 0
        self.force_random = 0
        self.dist_reached = False
        self.end_node = -1
        self.path = []
        self.graph_printed = False

        self.growth_factor = growth_factor
        self.e = e
        self.end_dist = end_dist
        self.rrt_star_rad = rrt_star_rad

        if not start_point.any():
            print("No start point")
        assert start_point.any()
        if not end_point.any():
            print("No end point")
        assert end_point.any()
        if not bin_map.any():
            print("No bin_map")
        assert bin_map.any()

    def find_closest(self, point, node_arr=np.array(False)):
        if not node_arr.any():
            node_arr = self.nodes
        nodes_tree = scipy.spatial.cKDTree(node_arr)
        return nodes_tree.query(point)

    def check_obstacle(self, point1, point2):
        shift_vector = (point2 - point1).astype(np.int32)
        iters = sum(abs(shift_vector))
        shift_vector = shift_vector / iters
        all_shift = shift_vector
        c_point = np.array(False)
        iters_made = 0
        for i in range(1, iters + 1):
            iters_made = i
            all_shift = np.copy(shift_vector * i)
            c_point = np.around(point1 + shift_vector * i).astype(np.int64)
            if self.bool_map[tuple(c_point)]:
                i-=1
                iters_made = i
                break
            if np.linalg.norm(shift_vector * i) >= self.growth_factor:
                break
        if np.linalg.norm(all_shift) < self.e or not c_point.any():
            return np.array(False), None, False
        if iters_made > 1:
            return c_point, np.linalg.norm(all_shift), iters_made==iters
        else:
            return np.array(False), None, False


    def step(self):
        node = self.nodes[self.node_num - 1]
        dist = np.linalg.norm(node - self.end_point)
        if self.force_random:
            self.add_random()
            self.force_random -= 1
        elif dist < self.end_dist and self.force_random == 0:
            self.add_near_end()
            self.force_random = 50
            if self.stuck > 10:
                self.stuck = 0
                self.force_random = 50
        else:
            self.add_random()



####### slow ##############
    def iter_node_map(self, node, curr_dim, pos):
        iter = node[0]
        mi = max(0, iter-self.rrt_star_rad)
        ma = min(iter+self.rrt_star_rad, self.map_shape[curr_dim])

        if node.shape[0] > 1:
            node = node[1:]
            curr_dim+=1
            for i in range(mi, ma):
                pos[curr_dim-1] = i
                self.iter_node_map(node, curr_dim, pos)
        else:
            for i in range(mi, ma):
                if self.node_map[(*pos, i)] != -1:
                    self.node_neighbours.append(self.node_map[(*pos, i)])

    #переписать на KDtree
    def check_node_region(self, new_node):
        node_neighbours = []
        nodes_copy = np.copy(self.nodes)
        for i in range(100):
            closest_node = self.find_closest(new_node, nodes_copy)
            nodes_copy[closest_node[1]] = np.ones(new_node.shape)*-1
            if closest_node[0] < self.rrt_star_rad:
                node_neighbours.append(closest_node[1])
            else:
                break


        #self.node_neighbours = []
        #curr_dim = 0
        #pos = [0]*(new_node.shape[0]-1)
        #self.iter_node_map(new_node, curr_dim, pos)
        return node_neighbours
############################################################

    def find_best_connection(self, new_node, neighbours):
        neighbours = [[i, self.nodes[i], *self.graph[i]] for i in neighbours]
        neighbours.sort(key=lambda x: x[-1])
        have_parent = False
        for i in neighbours:
            _, dist, reached = self.check_obstacle(new_node, i[1])
            if reached:
                if not have_parent:
                    self.nodes = np.append(self.nodes, [new_node], axis=0).astype(np.uint32)
                    self.graph[self.node_num] = [i[0], [], dist + self.graph[i[0]][2]]
                    self.node_map[tuple(new_node)] = self.node_num
                    self.graph[i[0]][1].append(self.node_num)
                    self.node_map[tuple(new_node)] = self.node_num
                    self.node_num += 1
                    have_parent = True
                else:
                    if dist + self.graph[self.node_num-1][2] < self.graph[i[0]][2]: #and self.graph[i[0]][0] != i[2]:
                        if i[0] in self.graph[i[2]][1]:
                            self.graph[i[0]][0] = self.node_num-1
                            self.graph[i[2]][1].remove(i[0])
                            self.graph[self.node_num - 1][1].append(i[0])
                            self.rebalance(i[0], self.graph[i[0]][2] - dist - self.graph[self.node_num-1][2])


    def rebalance(self, node_num, delta):
        self.graph[node_num][2] -= delta
        for i in self.graph[node_num][1]:
            self.rebalance(i, delta)


    def add_node_to_closest(self, new_node):
        closest_node = self.find_closest(new_node)
        node, dist, _ = self.check_obstacle(self.nodes[closest_node[1]], new_node)
        if node.any():
            if self.star:
                neighbors = self.check_node_region(node)
                self.find_best_connection(node, neighbors)
            else:
                self.nodes = np.append(self.nodes, [node], axis=0).astype(np.uint32)
                self.graph[self.node_num] = [closest_node[1], [], dist + self.graph[closest_node[1]][2]]
                self.node_map[tuple(node)] = self.node_num
                self.graph[closest_node[1]][1].append(self.node_num)
                self.node_num += 1

            return node
        else:
            return np.array(False)



    def add_random(self):
        random_point = (np.random.rand(len(self.map_shape))*self.map_shape).astype(np.uint32)
        self.random_point = random_point
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
            self.graph_printed = True
