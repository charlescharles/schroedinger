import pygame
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from numpy import nan
from schroedinger import S1

w, h = 800, 600
screen = pygame.display.set_mode((w, h))

draw_on = False
last_pos = (0, 0)
path = np.zeros(w)
color = (255, 128, 0)
radius = 10

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

try:
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.circle(screen, color, e.pos, radius)
            path[e.pos[0]] = 600 - e.pos[1]
            draw_on = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
            raise StopIteration
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos,  radius)
                path[e.pos[0]] = 600 - e.pos[1]
            last_pos = e.pos
        pygame.display.flip()

except StopIteration:
    pass

path = Series(np.trim_zeros(path)).replace(0, nan)
path = 0.01 * np.asarray(path.interpolate())
N = len(path)

#plt.plot(path)

dx = 0.01
xmax = (N/2)*dx
rg = (-xmax, xmax)
#N = int((rg[1] - rg[0])/dx)
solver = S1(dx, rg)
psi0 = np.zeros(N)
psi0[0] = 0
psi0[1] = 1
psi = solver.shooting(path, psi0)
plt.plot(psi)


plt.show()
pygame.quit()