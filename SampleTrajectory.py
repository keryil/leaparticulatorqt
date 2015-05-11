# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
import os

sys.path.append(os.path.expanduser("~/Dropbox/ABACUS/Workspace/Abacus/"))
sys.path.append(os.path.expanduser("~/Dropbox/ABACUS/Workspace/LeapArticulator"))

# <codecell>

from abacus.experiments.artificial.symbol_generator import SymbolGenerator
from abacus.experiments.artificial.trajectory import Trajectory
import numpy as np

np.seterr(all='raise')

# <codecell>

t = Trajectory(duration=36, step_size=4, prob_c=.25, dim_size=(100, 100), ndim=2, plot=False, width=0.002)

# <codecell>

colors = ['red', 'green', 'yellow', 'pink', 'navy', 'magenta', 'purple', 'pink', 'grey']
# width = 0.004
def show_trajectory(t, n_gen=2, generations_arr=None, width=0.003, noise=True):
    plt.quiverkey(t.plot2d(), 0.01, .98, 1, "Orig")
    # n_gen = 2
    t.plot2d(color="black", width=width)
    for i, c in zip(range(n_gen), colors):
        t_new = t
        if noise:
            t_new = t.noise(spread=1, in_place=False)
        q = t_new.plot2d(color=c, width=width)
        plt.quiverkey(q, i / float(n_gen) + .2, .98, 1, "Generation %d" % (int(i) + 1), coordinates='axes')
        if generations_arr is not None:
            generations_arr.append(t_new)
    plt.legend()


def plot_discretized(trajectory, symbol_generator, figure=1, color="blue", width=0.003, *args, **kwargs):
    discrete_path = [symbol_generator.codebook[i] for i in symbol_generator.generate(t.data)]
    print discrete_path
    x, y = zip(*discrete_path)
    x, y = np.array(x), np.array(y)
    plt.figure(figure)
    q = plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=color,
                   width=width, *args, **kwargs)
    return q

# <codecell>

width = 0.004
generations = []
plt.figure(0)
show_trajectory(t, n_gen=4, generations_arr=generations)
all_data = []
all_data.extend(t.data)
for g in generations:
    all_data.extend(g.data)

symbols = SymbolGenerator(all_data, 36)
# print all_data
print generations
x, y = [], []
for xi, yi, in symbols.codebook:
    x.append(xi)
    y.append(yi)
# x, y = list(x), list(y)
# plt.figure(1)
x, y = np.array(x), np.array(y)
plt.scatter(x, y, marker='x', color="blue", s=25)
q = plot_discretized(t, symbols, figure=1, color="black", alpha=.25, width=width)
plt.quiverkey(q, 0.01, .98, 1, "Orig")
for color, g in zip(colors, generations):
    i = colors.index(color)
    q = plot_discretized(g, symbols, figure=1, color=color, alpha=0.25, width=width)
    plt.quiverkey(q, i / float(len(generations)) + .2, .98, 1, "Generation %d" % (i + 1), coordinates='axes')
    print "Plotted generation %d" % (i + 1)

plt.figure(2)
show_trajectory(t, n_gen=0, width=width, noise=False)

# <codecell>

plt.figure(2)
t.plot2d()
plt.figure(2)
t.plot2d()

