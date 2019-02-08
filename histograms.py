import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import seaborn as sns
import pandas as pd


def hypersphere_sample(npoints, ndim=128):
    vec = np.random.uniform(low=-1, high=1, size=(npoints, ndim))
    for point in vec:
        point /= np.linalg.norm(point, axis=0)
    return vec


def hypersphere_sample_2(npoints, ndim=128):
    vec = np.zeros((npoints, ndim))
    for i in range(npoints):
        sum = 0
        vec[i][0] = np.random.uniform(low=-1, high=1)
        sum += vec[i][0]**2
        for j in range(1, ndim-1):
            vec[i][j] = np.random.uniform(low=-1 + sum, high=1 - sum)
            sum += vec[i][j] ** 2
        if bool(random.getrandbits(1)):
            vec[i][ndim - 1] = math.sqrt(abs(1 - sum))
        else:
            vec[i][ndim - 1] = -math.sqrt(abs(sum - 1))
        sum += vec[i][ndim - 1] ** 2

    for point in vec:
        point /= np.linalg.norm(point, axis=0)
        np.random.shuffle(point)
    return vec


def random_sphere_test(vec):
    df = pd.DataFrame(data=vec)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[0], df[1], zs=df[2], s=1)
    plt.show()


def sphere_display(npoints, ndim=3, animate=True, abs_color=True, density_color=False, frac=8, cmap='cool', alpha=1.0,
                   savepath=''):
    points = pd.DataFrame(hypersphere_sample_2(npoints, ndim))
    assert (len(points.columns) >= 3)
    plt.style.use('dark_background')
    fig = plt.figure()
    plt.gca().patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    title = ax.set_title('Test: Drop Sphere')
    if density_color:
        # fencepost makes it possible to count some points twice,  hence assert for sum being equal to points
        # should add warning that for >3 dimensions, density is only measuring density per voxel of casted space
        points['density'] = np.zeros(len(points.index))
        interval = 1 / (frac / 2)
        sum = 0
        for i in range(frac):
            ini = points.ix[((i * interval) - 1 <= points[0]) & (points[0] <= (i * interval + interval) - 1)]
            for j in range(frac):
                inj = ini.ix[((j * interval) - 1 <= ini[1]) & (ini[1] <= (j * interval + interval) - 1)]
                for k in range(frac):
                    ink = inj.ix[((k * interval) - 1 <= inj[2]) & (inj[2] <= (k * interval + interval) - 1)]
                    # density[i, j, k] = len(ink)
                    for idx in ink.index:
                        points.loc[idx, 'density'] = len(ink)
                    sum += len(ink)
        assert (sum == len(points.index))

        graph = ax.scatter(points[0], points[1], points[2], marker=",", s=1, c=points['density'], cmap=cmap,
                           alpha=alpha)

    else:
        if abs_color:
            if ndim > 3:
                graph = ax.scatter(points[0], points[1], points[2], marker=",", s=1, c=abs(points[3]), cmap=cmap)
            else:
                graph = ax.scatter(points[0], points[1], points[2], marker=",", s=1, c=abs(points[2]), cmap=cmap)

        else:
            if ndim > 3:
                graph = ax.scatter(points[0], points[1], points[2], marker=",", s=1, c=points[3], cmap=cmap)
            else:
                graph = ax.scatter(points[0], points[1], points[2], marker=",", s=1, c=points[2], cmap=cmap)

    global z
    z = points[2]

    def sphere_anim_fall(num):
        global z
        if num == 0:
            z = points[2]
        z = z - 0.001 * num ** 2
        for i in range(len(z)):
            if z[i] < -1:
                z[i] = -1
        graph._offsets3d = (points[0], points[1], z)
        title.set_text('Test: Drop {}-Sphere at t={}'.format(ndim, num))

    if animate:
        ani = matplotlib.animation.FuncAnimation(fig, sphere_anim_fall, 25, interval=40, blit=False)
    if savepath == '':
        plt.show()
    else:
        plt.savefig(savepath)


def random_circle_test(vec, draw_points=False):
    plt.style.use('dark_background')
    df = pd.DataFrame(data=vec)
    ax = sns.jointplot(x=0, y=1, data=df, kind="kde", cmap='gray')
    if draw_points:
        ax.plot_joint(plt.scatter, c="w", s=1, linewidth=1, marker=".")
        ax.ax_joint.collections[0].set_alpha(0)
    ax.set_axis_labels(str(0), str(1))
    plt.show()


def n_sphere_dataframe(highsphere, npoints):
    df = pd.DataFrame()
    for i in range(2, highsphere + 1):
        vec = np.random.randn(npoints, i)
        vec /= np.linalg.norm(vec, axis=0)
        nans = np.zeros((npoints, highsphere - i))
        # nans[:] = np.nan
        vec = np.append(vec, nans, axis=1)
        helper = pd.DataFrame(data=vec)
        df = df.append(helper, ignore_index=True)
    names = []
    for i in range(2, highsphere + 2):
        for j in range(0, npoints):
            names.append(str(i))
    df['Dimensions'] = pd.Series(data=names)
    return df


if __name__ == "__main__":
        sphere_display(3000, 4)
