import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#функция для выполнения поворота точек (X, Y) на угол Alpha относительно начала координат.
def rot2d(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY
#функция принимает две точки A и B и возвращает координаты для построения стрелки от A к B. Если A == B, стрелка отсутствует.
def get_arrow_coordinates(A, B):
    x0, y0 = A
    x1, y1 = B
    xarrow = np.array([-0.25, 0, -0.25])
    yarrow = np.array([-0.25, 0, 0.25])
    xarrow, yarrow = rot2d(xarrow, yarrow, math.atan2(y1 - y0, x1 - x0))
    if A == B:
        xarrow, yarrow = np.zeros(3), np.zeros(3)
    xarrow += x1
    yarrow += y1
    return np.concatenate([[x0, x1], xarrow]), np.concatenate([[y0, y1], yarrow])

#   определяем символ t и задаем формулы для r, phi, x, y, vx, vy, wx, и wy.
#   затем создаем массив временных шагов T.
#   далее идет цикл, в котором вычисляем значения координат, скоростей и ускорений для каждого времени t и сохраняем их в массивы X, Y, Vx, Vy, Wx, и Wy.
t = sp.Symbol("t")

r = 2 + sp.sin(8 * t)
phi = t + 0.2 * sp.cos(6 * t)
x = r * sp.cos(phi)
y = r * sp.sin(phi)

vx = sp.diff(x, t)
vy = sp.diff(y, t)
V = sp.sqrt(vx**2 + vy**2)
VxN = vx/V
VyN = vy/V
wx = sp.diff(vx, t)
wy = sp.diff(vy, t)
W = sp.sqrt(wx**2 + wy**2)
WxN = wx/W
WyN = wy/W

T = np.linspace(0, 10, 1000)
X = np.zeros_like(T)
Y = np.zeros_like(T)

Vx = np.zeros_like(T)
Vy = np.zeros_like(T)
Wx = np.zeros_like(T)
Wy = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    Vx[i] = sp.Subs(VxN, t, T[i])
    Vy[i] = sp.Subs(VyN, t, T[i])
    Wx[i] = sp.Subs(WxN, t, T[i])
    Wy[i] = sp.Subs(WyN, t, T[i])

#    Создается объект fig для рисования.
#    Создается ось ax1 для графика.
#    Устанавливаются аспекты и пределы осей.
#    Рисуется график X Y.
#    Определяются начальные координаты P.
#    Определяются графики для векторов anim_R0, anim_V, и anim_W.
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.plot(X, Y)

ax1.set(xlim=[-5, 5], ylim=[-7, 7])

P, = ax1.plot(X[0], Y[0], marker='o')
anim_R0, = ax1.plot(*get_arrow_coordinates([0, 0], [Vx[0], Vy[0]]),'y')
anim_V, = ax1.plot(*get_arrow_coordinates([0, 0], [X[0] + Vx[0], Y[0] + Vy[0]]),'r')
anim_W, = ax1.plot(*get_arrow_coordinates([0, 0], [X[0] + Wx[0], Y[0] + Wy[0]]),'g')

#   Функция anima обновляет данные для каждого кадра анимации.
#   P - точка, представляющая положение частицы.
#   anim_R0 - вектор радиус-вектора.
#   anim_V - вектор скорости.
#   anim_W - вектор ускорения.
def animation(i):
    P.set_data(X[i], Y[i])
    anim_R0.set_data(*get_arrow_coordinates([0, 0], [X[i], Y[i]]))
    anim_V.set_data(*get_arrow_coordinates([X[i], Y[i]], [X[i] + Vx[i], Y[i] + Vy[i]]))
    anim_W.set_data(*get_arrow_coordinates([X[i], Y[i]], [X[i] + Wx[i], Y[i] + Wy[i]]))
    return P, anim_R0, anim_V, anim_W

# cоздается анимация с использованием FuncAnimation, которая вызывает animation для каждого кадра анимации.
anim = FuncAnimation(fig, animation, frames=1000, interval=100, repeat=True)

plt.show()
