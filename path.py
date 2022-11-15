import pygame as pg
import numpy as np
import cv2

map_size = 150
pg.init()
map_arr = np.array([[[100, 100, 100]]*map_size]*map_size)
print(map_arr)
nav_arr = np.array([[0]*map_size]*map_size)
screen = pg.display.set_mode([500, 500])
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    screen.blit(pg.surfarray.make_surface(cv2.resize(map_arr, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)), (0, 0))


pg.quit()