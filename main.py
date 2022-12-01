from KUKA import KUKA
from Pygame_GUI.GUI_pygame import GuiControl
from map_plotter import MapPlotter
import threading as thr

# robot = KUKA('192.168.88.24', ros=False, offline=False)
# robot = KUKA('192.168.88.24', ros=True, offline=False)
robot = ['192.168.88.21', '192.168.88.22', '192.168.88.23', '192.168.88.24', '192.168.88.25']

# new_map = MapPlotter(robot)
# map_thr = thr.Thread(target=new_map.create_map, args=())
# map_thr.start()

sim = GuiControl(600, 400, robot)
sim.run()
