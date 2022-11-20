from KUKA import KUKA
from Pygame_GUI.GUI_pygame import GuiControl
from map_plotter import MapPlotter
import threading as thr
from path import RRT_sim

robot = KUKA('192.168.88.25', ros=False, offline=False)


new_map = MapPlotter(robot)
map_thr = thr.Thread(target=new_map.create_map, args=())
map_thr.start()
#rrt_sim = RRT_sim(robot.increment, new_map)
#rrt_thr = thr.Thread(target=rrt_sim.start, args=())
#rrt_thr.start()


sim = GuiControl(600, 400, robot)
sim.run()



