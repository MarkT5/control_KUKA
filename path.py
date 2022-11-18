import pygame as pg
import numpy as np
import cv2
from RRT import RRT


class RRT_sim:
    def __init__(self, curr_pos=None, plotter = None):
        self.start_point = curr_pos
        self.plotter = plotter
        self.map_height = 900
        self.map_width = 600
        self.disp_scale = 1
        self.discrete = 0.5


        pg.init()
        self.rrt = RRT()
        self.clock = pg.time.Clock()
        self.map_arr = np.array([[[255, 255, 255]] * (self.map_height + 1)] * (self.map_width + 1)).astype(np.uint8)
        # 1 = 5cm
        self.screen = pg.display.set_mode([self.map_width*self.disp_scale, self.map_height*self.disp_scale])
        self.running = True
        obstacle1 = [[81, 186], [73, 181], [131, 87], [142, 92]]
        obstacle1_conv = np.array(list(map(self.m_to_arr, obstacle1)), np.int32)
        obstacle2 = [[383, 187], [374, 193], [306, 94], [316, 87]]
        obstacle2_conv = np.array(list(map(self.m_to_arr, obstacle2)), np.int32)
        obstacle3 = [[207, 232], [181, 232], [181, 196], [207, 196]]
        obstacle3_conv = np.array(list(map(self.m_to_arr, obstacle3)), np.int32)

        cv2.rectangle(self.map_arr, (0, 0), (self.map_height, self.map_width), (0, 0, 0), 1)
        cv2.fillPoly(self.map_arr, pts=[obstacle1_conv, obstacle2_conv, obstacle3_conv], color=(0, 0, 0))

        self.tree_stage = False
        self.step = False
        self.flow = False

        self.end_point = None
        nav_map = []
        for i in self.map_arr:
            for j in i:
                if j[0] == 255:
                    nav_map.append(0)
                else:
                    nav_map.append(1)
        self.nav_map = np.array(nav_map)
        self.nav_map = self.nav_map.reshape(len(self.map_arr), len(self.map_arr[0]))

    # make map
    def m_to_arr(self, coords):
        x, y = coords
        return [x // self.discrete, y // self.discrete]

    def screen_to_arr(self, coords):
        x, y = coords
        return [x // self.disp_scale, y // self.disp_scale]

    def update_keys(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_s:
                    self.step = True
                    self.flow = False
                elif event.key == pg.K_d:
                    self.flow = True
                    self.step = True
                elif event.key == pg.K_a:
                    self.flow = False
                elif event.key == pg.K_SPACE:
                    self.tree_stage = True
            elif event.type == pg.MOUSEBUTTONDOWN:
                if not self.start_point:
                    self.start_point = self.screen_to_arr(pg.mouse.get_pos())
                    print("start point:", self.start_point)
                elif not self.end_point:
                    self.end_point = self.screen_to_arr(pg.mouse.get_pos())
                    print("end point:", self.end_point)

    def draw_map(self):
        self.screen.blit(
            pg.surfarray.make_surface(cv2.resize(self.map_arr, dsize=(self.map_height*self.disp_scale, self.map_width*self.disp_scale), interpolation=cv2.INTER_NEAREST)),
            (0, 0))
        if self.start_point:
            pg.draw.circle(self.screen, (255, 0, 0), list(map(lambda x: x * self.disp_scale, self.start_point)), 5)
        if self.end_point:
            pg.draw.circle(self.screen, (0, 255, 0), list(map(lambda x: x * self.disp_scale, self.end_point)), 5)

    def start(self):
        while self.running:
            self.update_keys()
            self.draw_map()
            # out pygame
            pg.display.update()
            pg.display.flip()
            self.clock.tick(10)
            if self.end_point and self.tree_stage:
                self.step_rrt()
        pg.quit()

    def step_rrt(self):
        self.rrt.start_point = np.array(self.start_point)
        self.rrt.end_point = np.array(self.end_point)
        if self.plotter:
            self.rrt.bool_map = self.plotter.map_arr
        self.rrt.bool_map = np.array(self.nav_map).astype(np.uint8)
        self.rrt.start()
        self.rrt.step()
        print("start RRT")
        while self.running:
            self.update_keys()
            self.draw_map()
            if self.step and not self.rrt.dist_reached:
                self.rrt.step()
            for j in range(self.rrt.nodes.shape[0]):
                i = self.rrt.nodes[j]
                pg.draw.circle(self.screen, (0, 0, 255), list(map(lambda x: x * self.disp_scale, i)), 5)
            pg.draw.circle(self.screen, (255, 0, 0), list(map(lambda x: x * self.disp_scale, self.start_point)), 5)
            for i in range(1, self.rrt.node_num):
                n = self.rrt.graph[i][0]
                pg.draw.aaline(self.screen, (255, 0, 255), list(map(lambda x: x * self.disp_scale, self.rrt.nodes[i])), list(map(lambda x: x * self.disp_scale, self.rrt.nodes[n])))
            pg.draw.circle(self.screen, (255, 0, 255), list(map(lambda x: x * self.disp_scale, self.rrt.random_point)),
                           5)
            if self.rrt.dist_reached:
                self.rrt.get_path()
                pg.draw.lines(self.screen, (255, 0, 0), False, [[*i] for i in list(map(lambda x: x * self.disp_scale, self.rrt.path))], 5)

            # out pygame
            pg.display.update()
            pg.display.flip()
            self.clock.tick(40)
            if not self.flow:
                self.step = False
        pg.quit()


#rrt_sim = RRT_sim()
#rrt_sim.start()
