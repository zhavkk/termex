
import math as mh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def rotate(x, y, theta):
    xr = mh.cos(theta) * x - mh.sin(theta) * y
    yr = (mh.sin(theta)) * x + np.cos(theta) * y
    return xr, yr


def Film(i):
    Point_A.set_data(X_A[i], Y_A[i])
    Point_B.set_data(X_B[i], Y_B[i])
    Box.set_data(X_A[i] + X_Box, Y_A[i] + Y_Box)
    Line_AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])
    return [Point_A, Point_B, Box, Line_AB]


steps = 400
t = np.linspace(10, 0, steps)
S = 5
phi = np.sin(t)
s = t
x = t

fig = plt.figure(figsize=[20, 8])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-3, 10], ylim=[-3, 10])

BoxX = 4
BoxY = 2
bx = BoxX / 2
by = BoxY / 2
l = 3
Pi = np.pi
alpha = Pi / 6
alphay = (55 * Pi) / 180
tg = np.tan(alpha)


# Платформы
X_Ground = [10 / np.tan(alphay), 10, -6, 10]
Y_Ground = [10, 8 * tg, -8 * tg, -8 * tg]
ax.plot(X_Ground, Y_Ground, color='black', linewidth=3)
#


# Брусок
x1, y1 = rotate(-bx, by, alpha)
x2, y2 = rotate(bx, by, alpha)
x3, y3 = rotate(bx, -by, alpha)
x4, y4 = rotate(-bx, -by, alpha)

X_Box = np.array([x1, x2, x3, x4, x1])
Y_Box = np.array([y1, y2, y3, y4, y1])
#


# Точки А и В
X_A = BoxX / 2 + x - S
Y_A = X_A * tg
X_B = X_A + l * np.sin(phi)
Y_B = Y_A - l * np.cos(phi)
#


Point_A = ax.plot(X_A[0], Y_A[0], marker='o')[0]
Point_B = ax.plot(X_B[0], Y_B[0], marker='o', markersize=20, color='black')[0]
Box = ax.plot(X_A[0] + X_Box, Y_A[0] + Y_Box, color='black', linewidth='2')[0]
Line_AB = ax.plot([X_A[0], X_B[0]], [Y_A[0], Y_B[0]], color='black', linewidth='2')[0]


anima = FuncAnimation(fig, Film, frames=steps, interval=10)
plt.show()
