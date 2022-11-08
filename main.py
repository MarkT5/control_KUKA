from KUKA import KUKA
from GUI_pygame import GuiControl
from map_plotter import MapPlotter
import threading as thr

robot = KUKA('192.168.88.25', ros=False, offline=False)


new_map = MapPlotter(robot)
map_thr = thr.Thread(target=new_map.create_map, args=())
map_thr.start()

sim = GuiControl(600, 400, robot)
sim.run()



