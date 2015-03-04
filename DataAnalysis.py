# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
os.chdir(os.path.expanduser("~/ThereminData"))
# !ls

# <markdowncell>

# # Import data and output to file

# <codecell>

from ExperimentalData import fromFile, toCSV
import pandas as pd
import jsonpickle
import numpy as np
pd.set_option("display.max_columns", None)
file_id = "1230105514.master"
filename_log = "logs/%s.exp.log" % file_id
responses, tests, responses_t, tests_t, images = fromFile(filename_log)
# toCSV(filename_log)

# <codecell>

def plot_quiver2d(data, alpha=.75, C=[], *args, **kwargs):
    X, Y = zip(*tuple(data))
    U = [x1-x0 for x0,x1 in zip(X[:-1],X[1:])]
    V = [y1-y0 for y0,y1 in zip(Y[:-1],Y[1:])]
    if C == []:
        color_delta = 1. / (len(X) - 1)
        C = [(color_delta*i,color_delta*i,color_delta*i) for i in range(len(X)-1)]
#     print C
    X, Y = X[:-1], Y[:-1]
#     print X, Y, U, V
    return quiver(X, Y, U, V, C, *args, scale_units='xy',angles='xy', scale=1, width=0.005, alpha=alpha, **kwargs)

def find_bounding_box(trajectories):
    xmin = ymin = 1000
    xmax = ymax = -1000
    delta = 1
    for signal in trajectories:
        for frame in signal:
            x, y, z = frame.get_stabilized_position()
            xmax = max(x + delta, xmax)
            xmin = min(x - delta, xmin)
            ymax = max(y + delta, ymax)
            ymin = min(y - delta, ymin)
    return xmin, xmax, ymin, ymax

def to_trajectory_object(trajectory, step_size=10):
    arr = [frame.get_stabilized_position()[:2] for frame in trajectory]
    t = Trajectory(from_arr=arr, duration=len(arr), step_size=step_size, origin=(xmin, ymin), 
                   ndim=2, dim_size=(xmax-xmin, ymax-ymin), prob_c=1)
    return t

def to_trajectory_file(trajectories, filename):
    xmin, xmax, ymin, ymax = find_bounding_box(trajectories)
    start = 0
    end = 1
    import os
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        print(xmin, xmax, ymin, ymax, start,end)
        f.write("%d %d %d %d %d %d\n" % (xmin, xmax, ymin, ymax, start,end))
        for signal in trajectories:
            for frame in signal:
                x, y, z = frame.get_stabilized_position()
                time = frame.timestamp
                f.write("%f %f %f\n" % (x, y, time))
            f.write("0.0 0.0 0.0\n")

# <codecell>

r = responses["127.0.0.1"]
# r = r['1']
# r = {"1":r}
def responses_to_trajectories(responses):
    counter = 0
    trajectories = []
    for phase in r:
        for image in r[phase]:
    #         if image == u'./img/meanings/5_1.png':
                counter+=1
                trajectory = r[phase][image]
                trajectories.append(trajectory)
                data = []
                for frame in trajectory[:-1]:
                    x, y, z = frame.get_stabilized_position()
                    data.append((x,y))
    #                 print frame.timestamp
    #             plot_quiver2d(data)
    #             break
    #             plot(X,Y,label="%s-%s" % (phase, image))
    return trajectories
trajectories = responses_to_trajectories(responses)
trajectories_t = responses_to_trajectories(responses_t)
to_trajectory_file(trajectories, "%s.trajectories" % (".".join(filename_log.split(".")[:-2])))

# <codecell>

from trajectory import Trajectory
all_trajectories = list(trajectories)
all_trajectories.extend(trajectories_t)
xmin, xmax, ymin, ymax = find_bounding_box(trajectories)
tr = [to_trajectory_object(trajectory, step_size=300) for trajectory in all_trajectories]

# <markdowncell>

# # K-means

# <codecell>

import sys,os
sys.path.append(os.path.expanduser("~/Dropbox/ABACUS/Workspace/Abacus"))
from abacus.experiments.artificial.symbol_generator import SymbolGenerator
colors = ['red','green','yellow','pink', 'navy', 'magenta', 'purple', 'blue', 'grey']

def plot_discretized(trajectory, symbol_generator, figure=1, color="blue", width=0.003, alpha=.5, *args, **kwargs):
    discrete_path = []
    for i in symbol_generator.generate(trajectory.data):
#         print i, symbol_generator.codebook[i]
        discrete_path.append(symbol_generator.codebook[i])
#     discrete_path = [tuple(symbol_generator.codebook[i]) for i in symbol_generator.generate(trajectory.data)]
#     print discrete_path
    x, y = zip(*discrete_path)
#     print any(x<0)
    x, y = np.array(x), np.array(y)
#     print any(x<0)
#     plt.figure(figure)
    scatter(x,y,alpha=alpha)
#     print map(tuple, discrete_path)
    q = plot_quiver2d(discrete_path, alpha)
#     q = plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, color=color, width=width, *args, **kwargs)
    return q


import numpy as np
from copy import deepcopy as copy
from random import shuffle
all = []
limit=None
tr_copy = copy(tr)
shuffle(tr_copy)
for t in tr_copy:
    all.extend(t.data)
symbol_generator = SymbolGenerator(all, 150)
# tr = tr[11:20]
f, axes = subplots(2,1, sharex=True, sharey=True)
sca(axes[0])
alpha = .2
# [plot_quiver2d(t, alpha=alpha) for t in tr[:limit]]
# sca(axes[1])
trn = []
for t in tr[:limit]:
    tn = t.noise(spread=50, in_place=False)
    trn.append(tn)
    plot_quiver2d(tn, alpha=alpha)
# print symbol_generator.generate(tr[0].data)
# codes = []
# for t in tr:
#     code = symbol_generator.generate(t.data)
#     codes.extend(tuple(map(tuple,symbol_generator.codebook[code])))
# print symbol_generator.codebook
plt.scatter(*zip(*tuple(symbol_generator.codebook)),color="red", marker="x")
sca(axes[1])
[plot_discretized(t,symbol_generator, color=c, alpha=alpha, figure=1,label="%s" % tr.index(t)) for c,t in zip(colors*5,tr[:limit])]
# sca(axes[3])
# [plot_discretized(t,symbol_generator, color=c, alpha=alpha, figure=1,label="%s" % trn.index(t)) for c,t in zip(colors*5,trn[:limit])]
gca().legend()

# <codecell>

data = np.column_stack(zip(*all))
print len(all)
from sklearn.cross_validation import train_test_split
train, test = train_test_split(data)
print len(train), len(test)
from IPython.parallel import Client
rc = Client()
lview = rc.load_balanced_view() # default load-balanced view
# print data
# print colors

# <markdowncell>

# #HMM Helper Functions

# <codecell>

from sklearn.hmm import GaussianHMM, GMMHMM
from sklearn.mixture import DPGMM

def fitDPGMM(n_components, data):
    from sklearn.mixture import DPGMM
    dpgmm = DPGMM(n_components=5, covariance_type='full')
    dpgmm.fit(data)
    score = dpgmm.score(data)
    return score, dpgmm
    # Run Gaussian HMM
# print("fitting to HMM and decoding ...")
    print("fitting to DPGMM and decoding ...")

def fitHMM(n_components, data):
    from sklearn.hmm import GaussianHMM

    # make an HMM instance and execute fit
#     print n_components, data[:4]
    model = GaussianHMM(n_components, covariance_type="tied", n_iter=1000)
    
    model.fit([data])
    score = model.score(data)
#     print score
    return score, model

def fitGMMHMM(n_components, data):
    from sklearn.hmm import GMMHMM

    # make an HMM instance and execute fit
#     print n_components, data[:4]
    model = GMMHMM(n_components, covariance_type="tied", n_iter=1000)
    
    model.fit([data])
    score = model.score(data)
#     print score
    return score, model

# <markdowncell>

# ## Gaussian HMM

# <codecell>

n_components = 6
alpha = .33
model = None
hidden_states = None
score = -9999999999
iterations = 1

lview.block = True
models_n_scores = lview.map(fitHMM, [n_components] * iterations , [data] * iterations)
# scores = lview.map(lambda x, data: x.score(data), models, [data] * iterations)
print models_n_scores
model = sorted(models_n_scores)[-1][1]

hidden_states = model.predict(data)
    # print("Hidden states")
    # for state in hidden_states:
    #     print state

print("done\n")

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
# print()

print("means and vars of each hidden state")
for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
#     print()

# <markdowncell>

# ## GMMHMM

# <codecell>

n_components = 4
alpha = .33
model = None
hidden_states = None
score = -9999999999
iterations = 20

lview.block = True
models_n_scores = lview.map(fitGMMHMM, [n_components] * iterations , [data] * iterations)
# scores = lview.map(lambda x, data: x.score(data), models, [data] * iterations)
print models_n_scores
model = sorted(models_n_scores)[-1][1]

hidden_states = model.predict(data)
    # print("Hidden states")
    # for state in hidden_states:
    #     print state

print("done\n")

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
# print()

print("means and vars of each hidden state")
for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model.gmms[i])
    print("var = ", np.diag(model.covars_[i]))
#     print()

# <markdowncell>

# ## Plotting HMMs

# <codecell>

from matplotlib.colors import colorConverter
from matplotlib.patches import Ellipse
# figure()
# means = []
# annotations = []

def on_pick(event):
    print event
    print annotations
    if event.artist in annotations:
        on_pick_annotation(event)
    elif event.artist in means:
        on_pick_means(event)
    draw()
    time.sleep(1)
    
def on_pick_annotation(event):
    print "Annotation:", event.artist
    event.artist.set_visible(False)

def on_pick_means(event):
    print "Mean:", means.index(event.artist)
    annotations[means.index(event.artist)].set_visible(True)
    print annotations[means.index(event.artist)]

colors = ['red','green','yellow', 'magenta', 'orange', 'black', 'cyan', 'white']
colors = [(x/10.,y/20.,z/30.) for x, y, z in zip(range(10), range(10), range(10))]
def plot_hmm(means_, transmat, covars, axes=None):
    if axes != None:
        axes(axes)
#     f, axes = subplots(2)#,sharex=True, sharey=True)
#     sca(axes[0])
    global annotations
    annotations = []
    global means
    means = []
    color_map = colors #[colorConverter.to_rgb(colors[i]) for i in range(len(means_))]
    for i in range(len(means_)):
        means.append(scatter(*tuple(means_[i]), color=colorConverter.to_rgb(colors[i]), picker=10, label="State%i"%i))
        annotate(s="%d" % i, xy=means[i], xytext=(1,-10), xycoords="data",textcoords="offset points", 
                         alpha=1,bbox=dict(boxstyle='round,pad=0.2', fc=colorConverter.to_rgb(colors[i]), alpha=0.3))
        gca().add_patch(Ellipse(xy = means_[i], width = np.diag(covars[i])[0], height = np.diag(covars[i])[1],
                        alpha=.15, color=colorConverter.to_rgb(colors[i])))
        x0, y0 = means_[i]
        prob_string = ""
        for j in range(len(means)):
            xdif = 10
            ydif = 5
            s = "P(%d->%d)=%f" % (i,j,transmat[i][j])
            prob_string = "%s\n%s" % (prob_string,s)
            if i != j:
                x1, y1 = means_[j]
                print transmat[i][j]
                # if transmat[i][j] is too low, we get an underflow here
#                 q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = 10000 * (transmat[i][j]**2),
                alpha = 10 ** -300
                if transmat[i][j] > 10 ** -100:
                    alpha = (100 * transmat[i][j])**2
                q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = alpha, 
                       scale_units='xy',angles='xy', scale=1, width=0.005, label="P(%d->%d)=%f" % (i,j,transmat[i][j]))
                legend()

        annotations.append(annotate(s=prob_string, xy=means_[i], xytext=(0, 10), xycoords="data",textcoords="offset points", 
                         alpha=1,bbox=dict(boxstyle='round,pad=0.2', fc=colorConverter.to_rgb(colors[i]), alpha=0.3), picker=True,
                         visible=False))


        print "State%i is %s" % (i, colors[i])
    cid = gcf().canvas.mpl_connect('pick_event', on_pick)
#     legend()
#     sca(axes[1])
#     print "Plot trajectories [%i of them]" % len(tr)
#     for t in tr:
#         print "trajectory: %s" % t
#         plot_quiver2d(t.data, C=color_map[:len(t.data)], alpha=alpha)
#         color_map = color_map[len(t.data):]
try:
    plot_hmm(model.means_, model.transmat_, model.covars_)
except:
    pass

# <codecell>

means = []
for i in range(n_components):
    means.append(scatter(*tuple(model.means_[i]), color=colorConverter.to_rgb(colors[i]), picker=10))
counter = 0
print tr
from collections import defaultdict
from matplotlib.patches import Arc
from math import atan, degrees, sqrt
import operator
for trajectory in tr[:]:
    trajectory_in_state_space = []
    state_counts = defaultdict(int)
    for x,y in trajectory.data:
        state = hidden_states[counter]
        state_counts[state] += 1
        trajectory_in_state_space.append(tuple(model.means_[state]))
        counter += 1
    state = max(state_counts.iteritems(), key=operator.itemgetter(1))[0]
    X, Y = zip(*tuple(trajectory_in_state_space))
#     print X
    color = colorConverter.to_rgb(colors[state])
    print color
    for i, (x,y) in enumerate(zip(X[:-1],Y[:-1])):
        y1 = Y[i+1]
        x1 = X[i+1]
        if (x != x1) or (y != y1): 
            rotation = degrees(arctan((y1-y)/(x1-x))) + 270
            print "Rotation is", rotation, " (%f,%f to %f,%f)" % (x,y,x1,y1)
            arc = Arc(xy=((x1 + x)/2., (y1 + y)/2.), width =  x1 - x, height =  sqrt((y1 - y) ** 2 + (x1-x) ** 2),
                      angle= rotation, theta1=90, theta2=270, alpha = .05)
            gca().add_patch(arc)
#             break
#     break
#         gca().plot(X,Y, marker=r'$\circlearrowleft$', alpha=1, color=color)
#     break

# <codecell>

# plot(*model.sample(10000), alpha=.25)
print file_id
model.fit([data])
model.score(data)

# <markdowncell>

# # Now for R

# <codecell>

# %reload_ext rpy2.ipython
import rpy2.robjects.numpy2ri
from rpy2.robjects import globalenv
inputForR = []
for trajectory in trajectories:
    inp = []
    inputForR.append(inp)
    for frame in trajectory:
        inp.append(list(frame.get_stabilized_position()))
inputForR = numpy.asarray(inputForR)

# <markdowncell>

# **Note:** The cell below includes functions to train multiple HMMs using the same parameters, and to pick the one with the lowest BIC. Because Baum-Welch algorithm is an EM algorithm, so it is easy to get stuck at local optima if only a few runs are made.
# 
# I tried to integrate this into ipcluster but for some reason source(blabla) doesn't work as expected, and even when it does, the returned HMM vector is a SexpVector instead of the expected ListVector.

# <codecell>

from IPython.parallel import Client
from functools import partial
rc = Client(profile='ssh')
lview = rc.load_balanced_view() # default load-balanced view
lview.block = True

def train_hmm_once(file_id, nstates, iter=1000):
    """
    Trains a Gaussian HMM using the data from an experimental log file, 
    using the specified number of states.
    """
#     from rpy2.robjects import r
    %load_ext rpy2.ipython
    %R source("~/Dropbox/ABACUS/Workspace/LeapArticulator/SampleHMM.R")
    %Rpush file_id 
    %Rpush nstates
    %R list[hmm, d] = fitHMM(file_id, nstates, iter=1000)
    %Rpull d
    %Rpull hmm
#     hmm, d = r("fitHMM(file_id, nstates, iter=1000)")
    print "Finished a %s state run" % nstates
    print "BIC:", hmm.rx("BIC")[0][0]
    print "AIC:", hmm.rx("AIC")[0][0]
    return hmm, d

def train_hmm_n_times(file_id, nstates, trials=20, iter=1000):
    """
    Trains multiple HMM's (as many as trials) and chooses the one with the 
    lowest BIC, so as to avoid local optima.
    """
    
    hmm, d, bic = None, None, None
    func = lambda args: train_hmm_once(file_id=args[0], nstates=args[1], iter=args[2])
    rc[:].push(dict(train_hmm_once=train_hmm_once))
    results = lview.map(func, [(file_id, nstates, 1000)] * trials)
    
    hmm, d, bic = None, None, 9999999999
    for hmmm, dd in results:
        # use this for ipcluster cases
        bbic = hmmm[2][0]
        # use this for non-cluster cases
        # bbic = hmmm.rx("BIC")[0][0]
        if bbic < bic:
            bic = bbic
            hmm = hmmm
            d = dd
    return hmm, d#, results
        
        

# <codecell>

from multiprocessing import Pool
pool = Pool()
print pool

# <codecell>

func = lambda args: train_hmm_n_times(file_id=args[0], nstates=args[1], trials=args[2], iter=args[3])
# lview.block = True
# nstates=6
trials = 20
args = [(file_id, nstates, trials, 1000) for nstates in range(5,26)]
results = map(func, args)# hmm, d, results = train_hmm_n_times(file_id, nstates, trials=20, iter=1000)
# pool.join()

# <codecell>

states, bics, aics = [], [], []
best = 0
for i, (hmm, d) in enumerate(results):
    if hmm is not None:
        nstates = hmm[0][2][1][0]
        bic = hmm[2][0]
        aic = hmm[3][0]
#         nstates = hmm.rx("HMM")[0].rx('distribution')[0].rx('nStates')[0]
#         bic = hmm.rx("BIC")[0][0]
#         aic = hmm.rx("AIC")[0][0]
        states.append(nstates)
        bics.append(bic)
        if min(bics) == bic:
            best = i
        aics.append(aic)
        #print aic, bic, nstates
#print states, bics
n = sum(map(len, trajectories))
# n = len(trajectories)
aicc = [aic + 2*k*(k+1)/(n-k-1) for aic, k in zip(aics, [s + s + s*s + s + s*2 for s in states])]
# plot nStates against BIC
scatter(states, bics, label="BIC", color="r")
scatter(states, aics, label="AIC", color='g')
# scatter(states, aicc, label="AICc", color='b')
legend()
hmm, d = results[best]
print best

# <codecell>

nstates=6
hmm, d = train_hmm(file_id, nstates, iter=1000)
bic = hmm.rx("BIC")[0][0]
for i in range(15):
    hmmm, dd = train_hmm(file_id, nstates, iter=1000)
    bbic = hmmm.rx("BIC")[0][0]
    if bbic < bic:
        print("Improvement: %f -> %f (delta: %f)" % (bic, bbic, bic - bbic))
        hmm = hmmm
        d = dd
        bic = bbic

# <codecell>

%%R
variances = hmm$HMM$distribution$cov
transmat = hmm$HMM$transMat
means = hmm$HMM$distribution$mean
path = list()
for(trajectory in d){
    path = c(path, viterbi(hmm, trajectory))
}

# <codecell>

means = [(m[0],m[1]) for m in list(hmm[0][2][3])]
nstates = len(means)
variances = [(tuple(v[:2]), tuple(v[2:])) for v in hmm[0][2][4]]
initProb = list(hmm[0][0])
transmat = list(hmm[0][1])
transmat_n = [[0] * nstates for i in range(nstates)]
print nstates
for i, n in enumerate(transmat):
    state_1 = i % nstates
    state_2 = i / nstates
    transmat_n[i % nstates][i / nstates] = n
transmat = transmat_n
print means

# <codecell>

import rpy2.robjects.numpy2ri

# %Rpull means d hmm path
# %Rpull variances transmat 
# print variances
paths = [numpy.asarray(path[i], dtype=int) for i in range(0, len(path), 3)]
# print means
# print d

# means__ = [[m for m in mean] for mean in means]
x = zip(*means)[0]
y = zip(*means)[1]
# scatter(x,y,color="red")
# print d
legends = []
items = []
ax = None
colors = ['red','green','yellow', 'magenta', 'orange', 'black', 'cyan', 'white'] * 3

# for i, (trajectory, p) in enumerate(zip(tr, paths)): 
#     print i
# #     for state in p:
# #         print "State:", state
#     tr_colors = [(colors*4)[int(state)-1] for state in p]
#     t = trajectory.plot2d(color=tr_colors)
#     legends.append("Trajectory%i" % i)
# #     items.append(p)
#     items.append(t)
#     #gca().legend()


plot_hmm(numpy.asarray(means__), 
        numpy.asarray(transmat), 
        variances, axes=ax)

# Let's create checkboxes
rax = plt.axes([0.05, 0.4, 0.1, 0.15])
# rax = plt.gca()
check = CheckButtons(rax, legends, [True] * len(legends))
# plt.sca(axes)

def func(label):
    widget = items[legends.index(label)]
    widget.set_visible(not widget.get_visible())
    plt.draw()
    
check.on_clicked(func)
plt.draw()
plt.show()
# print legends, items
#     break
# plot_quiver2d()
# scatter(d['x'],d['y'], alpha=.5)
# print numpy.asarray(transmat)

# <codecell>

print hmm

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(6*np.pi*t)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False, lw=2)
l1, = ax.plot(t, s1, lw=2)
l2, = ax.plot(t, s2, lw=2)
plt.subplots_adjust(left=0.2)

rax = plt.axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))

def func(label):
    if label == '2 Hz': l0.set_visible(not l0.get_visible())
    elif label == '4 Hz': l1.set_visible(not l1.get_visible())
    elif label == '6 Hz': l2.set_visible(not l2.get_visible())
    plt.draw()
check.on_clicked(func)

plt.show()

# <rawcell>


