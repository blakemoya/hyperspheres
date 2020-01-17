import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


t = np.array([[2, 0, 0],
              [0, 0.5, 0],
              [0, 0, 2]])
b = np.array([[1, 1, 0],
              [0, 1, 1],
              [1, 0, 1]])

lim = 3
scales = [min(lim / np.abs(b[:, i] + 0.00001)) for i in range(3)]
cols = ['red', 'green', 'blue']
max_frame = 240
turn = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.w_xaxis.set_pane_color((1, 0, 0, 0.1))
ax.w_yaxis.set_pane_color((0, 1, 0, 0.1))
ax.w_zaxis.set_pane_color((0, 0, 1, 0.1))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

lines = []

def init():
    global t
    global b
    global turn
    global lines
    lines = []
    lines.append(ax.plot([-lim, lim], [0, 0], [0, 0], ls='dashed', lw=0.75, alpha=0.25, c='black')[0])
    lines.append(ax.plot([0, 0], [-lim, lim], [0, 0], ls='dashed', lw=0.75, alpha=0.25, c='black')[0])
    lines.append(ax.plot([0, 0], [0, 0], [-lim, lim], ls='dashed', lw=0.75, alpha=0.25, c='black')[0])
    lines.append(ax.plot([0, 1], [0, 0], [0, 0], lw=0.75, c=cols[0])[0])
    lines.append(ax.plot([0, 0], [0, 1], [0, 0], lw=0.75, c=cols[1])[0])
    lines.append(ax.plot([0, 0], [0, 0], [0, 1], lw=0.75, c=cols[2])[0])
    for i in range(3):
        st = np.hstack((np.zeros((3, 1)), b[:, i, np.newaxis]))
        lines.append(ax.plot(st[0],
                             st[1],
                             st[2],
                             c=cols[i])[0])
    for i in range(3):
        st = np.hstack((-b[:, i, np.newaxis], b[:, i, np.newaxis]))
        lines.append(ax.plot(st[0] * scales[i],
                             st[1] * scales[i],
                             st[2] * scales[i],
                             ls='dashed',
                             alpha=0.5,
                             c=cols[i])[0])
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        st = np.hstack((b[:, j, np.newaxis],
                        b[:, j, np.newaxis] +
                        b[:, i, np.newaxis]))
        lines.append(ax.plot(st[0],
                            st[1],
                            st[2],
                            ls='dashed',
                            alpha=0.5,
                            c=cols[i])[0])
        st = np.hstack((b[:, k, np.newaxis],
                        b[:, k, np.newaxis] +
                        b[:, i, np.newaxis]))
        lines.append(ax.plot(st[0],
                            st[1],
                            st[2],
                            ls='dashed',
                            alpha=0.5,
                            c=cols[i])[0])
        st = np.hstack((b[:, j, np.newaxis] +
                        b[:, k, np.newaxis],
                        b[:, j, np.newaxis] +
                        b[:, k, np.newaxis] +
                        b[:, i, np.newaxis]))
        lines.append(ax.plot(st[0],
                            st[1],
                            st[2],
                            ls='dashed',
                            alpha=0.5,
                            c=cols[i])[0])
    return lines

def update(num):
    global lines
    global turn
    if num == max_frame / 2 or num == max_frame - 1:
        turn = not turn
    if turn:
        t_ = (t - np.diag([1, 1, 1])) * num / (max_frame / 2) + np.diag([1, 1, 1])
    else:
        t_ = (t - np.diag([1, 1, 1])) * (max_frame - num) / (max_frame / 2) + np.diag([1, 1, 1])
    b_ = b @ t_
    scales = [min(lim / np.abs(b_[:, i] + 0.00001)) for i in range(3)]
    for idx in range(6, 9):
        i = idx - 6
        st = np.hstack((np.zeros((3, 1)),
                        b_[:, i, np.newaxis]))
        lines[idx].set_data_3d(st[0],
                               st[1],
                               st[2])
    for idx in range(9, 12):
        i = idx - 9
        st = np.hstack((-b_[:, i, np.newaxis],
                        b_[:, i, np.newaxis]))
        lines[idx].set_data_3d(st[0] * scales[i],
                               st[1] * scales[i],
                               st[2] * scales[i])
    for idx in range(12, 20, 3):
        i = (idx - 12) // 3
        j = (i + 1) % 3
        k = (i + 2) % 3
        st = np.hstack((b_[:, j, np.newaxis],
                        b_[:, j, np.newaxis] +
                        b_[:, i, np.newaxis]))
        lines[idx].set_data_3d(st[0],
                               st[1],
                               st[2])
        st = np.hstack((b_[:, k, np.newaxis],
                        b_[:, k, np.newaxis] +
                        b_[:, i, np.newaxis]))
        lines[idx + 1].set_data_3d(st[0],
                                   st[1],
                                   st[2])
        st = np.hstack((b_[:, j, np.newaxis] +
                        b_[:, k, np.newaxis],
                        b_[:, j, np.newaxis] +
                        b_[:, k, np.newaxis] +
                        b_[:, i, np.newaxis]))
        lines[idx + 2].set_data_3d(st[0],
                                   st[1],
                                   st[2])
    return lines

ani = animation.FuncAnimation(fig, update, init_func=init, frames=max_frame, interval=5, blit=True)
plt.grid(b=None)
plt.show()