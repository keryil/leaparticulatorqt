# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Import data and output to file

# <codecell>

import ExperimentalData
reload(ExperimentalData)
from ExperimentalData import toCSV
import pandas as pd
import jsonpickle
import numpy as np
from leaparticulator import constants

colors = [(x/10.,y/20.,z/40.) for x, y, z in zip(range(10), range(10), range(10))]
colors.extend([(x/40.,y/20.,z/10.) for x, y, z in zip(range(1,10), range(1,10), range(1,10))])
colors.extend([(x/40.,y/10.,z/20.) for x, y, z in zip(range(1,10), range(1,10), range(1,10))])
# colors.extend(['red','green','yellow', 'magenta', 'orange', 'black', 'cyan', 'white'])

pd.set_option("display.max_columns", None)
# file_id = "1230105514.master"
# id_to_log = lambda x: "logs/%s.exp.log" % x

def print_n_flush(*args):
#     from __future__ import print_function
    import sys
    print_on_single_line = False
    for arg in args:
        if "\n" in arg:
            if print_on_single_line:
                print "\n"
                print_on_single_line = False
            for aa in arg.split("\n"):
                print aa
        else:
            print arg,
            print_on_single_line = True
    if print_on_single_line:
        print ""
    sys.stdout.flush()
    sys.stderr.flush()
# filename_log = id_to_log(file_id)
# responses, tests, responses_t, tests_t, images = toCSV(filename_log)

# <codecell>

# quiver_annotations = []
from trajectory import Trajectory
def plot_quiver2d(data, alpha=.75, C=[], path=None, *args, **kwargs):
    global quiver_annotations
    
    X, Y = zip(*tuple(data))
    U = [x1-x0 for x0,x1 in zip(X[:-1],X[1:])]
    V = [y1-y0 for y0,y1 in zip(Y[:-1],Y[1:])]
    if C == []:
        color_delta = 1. / (len(X) - 1)
        C = [(color_delta*i,color_delta*i,color_delta*i) for i in range(len(X)-1)]
#     print_n_flush(_n_flush( C))
    X, Y = X[:-1], Y[:-1]
#     print_n_flush(_n_flush( X, Y, U, V))
    patches = quiver(X, Y, U, V, *args, color=C, edgecolors=["black" for i in C], scale_units='xy',angles='xy', scale=1, width=0.005, alpha=alpha, **kwargs)
    return patches    
    
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

def to_trajectory_object(trajectory, xmin, ymin, xmax, ymax, step_size=10):
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
        print_n_flush(_n_flush((xmin, xmax, ymin, ymax, start,end)))
        f.write("%d %d %d %d %d %d\n" % (xmin, xmax, ymin, ymax, start,end))
        for signal in trajectories:
            for frame in signal:
                x, y, z = frame.get_stabilized_position()
                time = frame.timestamp
                f.write("%f %f %f\n" % (x, y, time))
            f.write("0.0 0.0 0.0\n")

# <codecell>

# r = responses["127.0.0.1"]
# r = r['1']
# r = {"1":r}
def responses_to_trajectories(responses):
    counter = 0
    trajectories = []
    for host in responses:
        r = responses[host]
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
        #                 print_n_flush( frame.timestamp)
        #             plot_quiver2d(data)
        #             break
        #             plot(X,Y,label="%s-%s" % (phase, image))
    return trajectories

# <markdowncell>

# ## Plotting HMMs

# <codecell>

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    from matplotlib.pyplot import gca
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(np.asarray(cov).reshape((2,2)))
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, ec="black", **kwargs)

    ax.add_artist(ellip)
    return ellip

# <codecell>

from matplotlib.colors import colorConverter
from matplotlib.patches import Ellipse
from matplotlib.pyplot import scatter, annotate, quiver, legend, gcf
from numpy import log
# figure()
# means = []
# annotations = []

def on_pick(event):
    print_n_flush( event)
    print_n_flush( annotations)
    if event.artist in annotations:
        on_pick_annotation(event)
    elif event.artist in means:
        on_pick_means(event)
    draw()
    time.sleep(1)

def on_pick_trajectory_event(event):
    pass
    
def on_pick_annotation(event):
    print_n_flush( "Annotation:", event.artist)
    event.artist.set_visible(False)

def on_pick_means(event):
    print_n_flush( "Mean:", means.index(event.artist))
    annotations[means.index(event.artist)].set_visible(True)
    print_n_flush( annotations[means.index(event.artist)])

# colors = ['red','green','yellow', 'magenta', 'orange', 'black', 'cyan', 'white']
def plot_hmm(means_, transmat, covars, initProbs, axes=None, clr=None, transition_arrows=True):
    if axes != None:
        axes(axes)
#     f, axes = subplots(2)#,sharex=True, sharey=True)
#     sca(axes[0])
    global annotations
    annotations = []
    global means
    means = []
    colors=clr
#     color_map = colors #[colorConverter.to_rgb(colors[i]) for i in range(len(means_))]
    for i, mean in enumerate(means_):
#         print_n_flush( "MEAN:", tuple(mean))
        means.append(scatter(*tuple(mean), color=colorConverter.to_rgb(colors[i]), picker=10, label="State%i"%i))
        annotate(s="%d" % i, xy=mean, xytext=(-10,-10), xycoords="data",textcoords="offset points", 
                         alpha=1,bbox=dict(boxstyle='round,pad=0.2', fc=colorConverter.to_rgb(colors[i]), alpha=0.3))
#         gca().add_patch(Ellipse(xy = means_[i], width = np.diag(covars[i])[0], height = np.diag(covars[i])[1],
#                         alpha=.15, color=colorConverter.to_rgb(colors[i])))
        plot_cov_ellipse(covars[i], mean, alpha=.15, color=colorConverter.to_rgb(colors[i]))
        x0, y0 = mean
        prob_string = "P(t0)=%f" % initProbs[i]
        for j, p in enumerate(transmat[i]):
            xdif = 10
            ydif = 5
            s = "P(%d->%d)=%f" % (i,j,p)
#             print_n_flush( "State%d: %s" % (i, s))
            prob_string = "%s\n%s" % (prob_string,s)
            if transition_arrows:
                if i != j:
                    x1, y1 = means_[j]
                    # if transmat[i][j] is too low, we get an underflow here
    #                 q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = 10000 * (transmat[i][j]**2),
                    alpha = 10 ** -300
                    if p > 10 ** -100:
                        alpha = (100 * p)**2
                    q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = 1 / log(alpha), 
                           scale_units='xy',angles='xy', scale=1, width=0.005, label="P(%d->%d)=%f" % (i,j,p))
#         legend()

        annotations.append(annotate(s=prob_string, xy=mean, xytext=(0, 10), xycoords="data",textcoords="offset points", 
                         alpha=1,bbox=dict(boxstyle='round,pad=0.2', fc=colorConverter.to_rgb(colors[i]), alpha=0.3), picker=True,
                         visible=False))


#         print_n_flush( "State%i is %s" % (i, colors[i]))
    cid = gcf().canvas.mpl_connect('pick_event', on_pick)

# <codecell>

def plot_hmm_path(trajectory_objs, paths, legends=[], items=[]):
    global colors
    print_n_flush( "Colors are:", colors)
    for i, (trajectory, p) in enumerate(zip(trajectory_objs, paths)): 
        print_n_flush( "Path:", p)
        tr_colors = [colors[int(state)] for state in p]
        t = trajectory.plot2d(color=tr_colors)
    #     t = plot_quiver2d(trajectory, color=tr_colors, path=p)
        too_high = [tt for tt in trajectory if tt[1] > 400]
#         print_n_flush( "Too high", too_high)
        legends.append("Trajectory%i" % i)
    #     items.append(p)
        items.append(t)
    #gca().legend()



    # Let's create checkboxes
    rax = plt.axes([0.05, 0.4, 0.1, 0.15])
    # rax = plt.gca()
    from matplotlib.widgets import CheckButtons
    check = CheckButtons(rax, legends, [True] * len(legends))
    # plt.sca(axes)

    def func(label):
        widget = items[legends.index(label)]
        widget.set_visible(not widget.get_visible())
        plt.draw()

    check.on_clicked(func)

# <markdowncell>

# # Now for R

# <markdowncell>

# **Note:** The cell below includes functions to train multiple HMMs using the same parameters, and to pick the one with the lowest BIC. Because Baum-Welch algorithm is an EM algorithm, so it is easy to get stuck at local optima if only a few runs are made.
# 
# I tried to integrate this into ipcluster but for some reason source(blabla) doesn't work as expected, and even when it does, the returned HMM vector is a SexpVector instead of the expected ListVector.

# <codecell>

def fn(args):
    from ExperimentalData import reduce_hmm
    from GHmmWrapper import train_hmm_on_set_of_obs
#     import dill
    data,nstates,range_x,range_y = args
    hmm = train_hmm_on_set_of_obs(data,nstates,range_x,range_y)
#     print type(hmm)
#     return 1
    return reduce_hmm(hmm)[1]
        
def train_hmm_n_times(file_id, nstates, trials=20, iter=1000, pickle=True, 
                      phase=2, cond=None, units=constants.XY, parallel=True):
    """
    Trains multiple HMM's (as many as trials parameter per nstate) and chooses the one with the 
    lowest BIC, so as to avoid local optima. units parameter can be "xy", "amp_and_freq", or
    "amp_and_mel", which specifies the kind of data to fit the HMM to. 
    """
    def pick_lowest_bic(models):
        hmm, d, bic = None, None, 9999999999
        for hmm_ in models:
#             hmm_ = HMM(hmm__, training_data=hmm__.obs, hmm_type="ghmm")
            if hmm_.bic < bic:
                bic = hmm_.bic
                hmm = hmm_
        if hmm is None:
            raise Exception("There are no valid models, WTF?!?")
#             return None
#         Hmm = HMM(hmm, training_data=d, hmm_type="hmmlearn")
#         print_n_flush( "New hmm and data (%s)" % d)
#         Hmm.from_R(hmm)
        return hmm
    
    
    
    import GHmmWrapper
    reload(GHmmWrapper)
    from GHmmWrapper import get_range_of_multiple_traj
#     reload(ExperimentalData)
    from ExperimentalData import fromFile, reconstruct_hmm
    from LeapTheremin import palmToAmpAndFreq,palmToAmpAndMel
    
    
    
    responses, test_results, responses_p, test_p, images = fromFile(id_to_log(file_id))
    multivariate = False
    reverse_cond = cond in ("2r","1r")
    interval = 1
    pick_var = 0
    if reverse_cond:
        interval = -1
        pick_var = 1
        
    if cond in ("2","2r"):
        if phase == 1:
            multivariate = True
    else:
        if phase == 2:
            multivariate = True
            
    formatData = None
            
    if multivariate:
        if units == constants.XY:
            formatData = lambda r, phase: [[frame.get_stabilized_position()[:2][::interval] for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
        elif units == constants.AMP_AND_FREQ:
            # -interval, because amp_and_freq returns y,x and not x,y. 
            formatData = lambda r, phase: [[palmToAmpAndFreq(frame.get_stabilized_position())[::-interval] for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
        elif units == constants.AMP_AND_MEL:
            # -interval, because amp_and_freq returns y,x and not x,y. 
            formatData = lambda r, phase: [[palmToAmpAndMel(frame.get_stabilized_position())[::-interval] for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
    else:
        if units == constants.XY:
            formatData = lambda r, phase: [[frame.get_stabilized_position()[pick_var] for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
        elif units == constants.AMP_AND_FREQ:
            # -interval, because amp_and_freq returns y,x and not x,y. 
            formatData = lambda r, phase: [[palmToAmpAndFreq(frame.get_stabilized_position())[::-interval][pick_var] for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
        elif units == constants.AMP_AND_MEL:
            # -interval, because amp_and_freq returns y,x and not x,y. 
            formatData = lambda r, phase: [[palmToAmpAndMel(frame.get_stabilized_position())[::-interval][pick_var] for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
    
    data = formatData(responses,phase) + formatData(responses_p,phase)
    print_n_flush("Sample data: %s" % data[0][:3])
#     data = [[frame.get_stabilized_position()[:2] for frame in response] for response in data]
#     data.append()
    lview=client=None
    if parallel:
        from IPython.parallel import Client
        client = Client(profile="default")
        from types import FunctionType
        from IPython.utils.pickleutil import can_map

        can_map.pop(FunctionType, None)
        import pickle
        from IPython.kernel.zmq import serialize
        serialize.pickle = pickle

        client[:].use_dill()
        reg ="import copy_reg, ExperimentalData;copy_reg.constructor(ExperimentalData.reconstruct_hmm);copy_reg.pickle(ExperimentalData.HMM, ExperimentalData.reduce_hmm, ExperimentalData.reconstruct_hmm)"
    #     print type(data), type(data[0])

        client[:].execute(reg)
        #     print data 

        lview = client.load_balanced_view() # default load-balanced 

        lview.block = True
    to_return = []
    range_x, range_y=get_range_of_multiple_traj(data)
    
    for n in nstates:
        print_n_flush("Doing %d state models..." % n) 
        args = [(data,n,range_x,range_y)] * trials
        
        if not parallel:    
            hmms = map(fn,args)#[(data,nstates,range_x,range_y)] * trials)
        else:
            hmms = lview.map(fn,args)#[(data,nstates,range_x,range_y)] * trials)
        hmms = [reconstruct_hmm(matrix, data) for matrix,data in hmms]
        
        to_return.append(pick_lowest_bic(hmms))

    if pickle:
        pickle_results(to_return, nstates, trials, iter, id_to_log(file_id), phase, units=units)
    return to_return
        
def pickle_results(results, nstates, trials, iter, filename_log, phase=None, units=constants.XY):
    import ExperimentalData
    hmms, ds = zip(*[ExperimentalData.reduce_hmm(hmm)[1] for hmm in results])
    assert any(hmms)
    assert any(ds)
#     print_n_flush(hmms)
#     results_pickled = jsonpickle.encode((hmms, ds, nstates, trials, iter))
    extension = ".hmms"
    if (phase is not None):
        extension = ".phase%d.%s.hmms" % (phase, units)
    print_n_flush( "Writing results to %s" % (filename_log + extension) )
    with open(filename_log + extension, "w") as f:
        for i, item in zip(("hmms", "ds", "nstates", "trials", "iter"),(hmms, ds, nstates, trials, iter)):
            print_n_flush( i)
#                 for a, hmm in enumerate(item):
#                     print_n_flush( a, "********\n", str(hmm))
# #                     import pickle
# #                     pickle.dumps(hmm)
# #                     encoded = jsonpickle.encode(hmm)
# #                     jsonpickle.decode(encoded)
#                 f.write(jsonpickle.encode(item))
#             else:
#             print_n_flush( item)
            f.write(jsonpickle.encode(item))
            f.write("\n")

def unpickle_results(filename_log, phase=None, units=None):
    import ExperimentalData
    from collections import namedtuple
    extension = ".hmms"
    hmms = ds = nstates = trials = iter = None
    if phase is not None:
        if units is not None:
            extension = ".phase%d.%s.hmms" % (phase, units)
        # this clause is purely for backward compatibility with
        # old pickle files
        else:
            extension = ".phase%d.hmms" % (phase)
    with open(filename_log + extension, "r") as f:
        hmms =  jsonpickle.decode(f.readline().rstrip())
        ds =  jsonpickle.decode(f.readline().rstrip())
#         print ds
        hmms = [ExperimentalData.reconstruct_hmm(hmm, d) for hmm,d in zip(hmms,ds)]
        nstates =  jsonpickle.decode(f.readline().rstrip())
        trials =  jsonpickle.decode(f.readline().rstrip())
        iter =  jsonpickle.decode(f.readline().rstrip())
    Results = namedtuple("Results", "hmms ds nstates trials iterations")
    return Results(hmms, ds, nstates, trials, iter)

# <codecell>

def responses_to_traj_objs(responses, responses_t):
    import trajectory
    # reload(trajectory)
    trajectories = responses_to_trajectories(responses)
    trajectories_t = responses_to_trajectories(responses_t)
    to_trajectory_file(trajectories, "%s.trajectories" % (".".join(filename_log.split(".")[:-2])))

    all_trajectories = list(trajectories)
    all_trajectories.extend(trajectories_t)
    xmin, xmax, ymin, ymax = find_bounding_box(trajectories)
    tr = [to_trajectory_object(trajectory, step_size=300, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) for trajectory in all_trajectories]
    return tr, trajectories

def pick_hmm_by_bic(hmms, responses, responses_t, plot=True):
    responses_to_traj_objs(responses, responses_t)
    states, bics, aics = [], [], []
    best = 0
    tr, trajectories = responses_to_traj_objs(responses, responses_t)
    for i, (hmm, d) in enumerate(results):
        if hmm is not None:
    #         nstates = hmm[0][2][1][0]
    #         bic = hmm[2][0]
    #         aic = hmm[3][0]
    #         nstates = hmm.rx("HMM")[0].rx('distribution')[0].rx('nStates')[0]
    #         bic = hmm.rx("BIC")[0][0]
    #         aic = hmm.rx("AIC")[0][0]
            states.append(hmm.nstates)
            bics.append(hmm.bic)
            if min(bics) == hmm.bic:
                best = i
            aics.append(hmm.aic)
            #print_n_flush( aic, bic, nstates)
#     print_n_flush( states)
    n = sum(map(len, trajectories))
    # n = len(trajectories)
    # aicc = [aic + 2*k*(k+1)/(n-k-1) for aic, k in zip(aics, [s + s + s*s + s + s*2 for s in states])]
    # plot nStates against BIC
    if plot:
        scatter(states, bics, label="BIC", color="r")
        scatter(states, aics, label="AIC", color='g')
        # scatter(states, aicc, label="AICc", color='b')
        legend()
    hmm, d = results[best]
#     print_n_flush( best)
#     print_n_flush( aics)
    return hmm, d

# <codecell>

def analyze_log_file(file_id, nstates, trials, iter):
#     id_to_log = lambda x: "logs/%s.exp.log" % x
    filename_log = id_to_log(file_id)
    responses, tests, responses_t, tests_t, images = toCSV(filename_log)
    
    from IPython.parallel import Client
#     from functools import partial
    from rpy2.rinterface import initr
    try:
        rinterface.set_initoptions(("--max-ppsize=500000"))
    except RuntimeError, e:
        print_n_flush( "Runtime error, probably redundant call to set_initoptions()")
        print_n_flush( e)
    initr()
    client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
    lview = client.load_balanced_view() # default load-balanced view
    lview.block = True
    # func = lambda args: train_hmm_n_times(file_id=args[0], nstates=args[1], trials=args[2], iter=args[3])
    # trials = 4
    client[:].push(dict(train_hmm_once=train_hmm_once))
    # args = [(file_id, nstates, trials, 1000) for nstates in range(5,10)]
    # results = lview.map(func, args)# hmm, d, results = train_hmm_n_times(file_id, nstates, trials=20, iter=1000)
    # pool.join()
    results = train_hmm_n_times(file_id, nstates=nstates, trials=trials, iter=iter)
    return results

def analyze_log_file_in_phases(file_id, nstates, trials, iter):
    print_n_flush( "Starting phase by phase analysis...")
#     id_to_log = lambda x: "logs/%s.exp.log" % x
    filename_log = id_to_log(file_id)
    responses, tests, responses_t, tests_t, images = toCSV(filename_log)
    from IPython.parallel import Client
#     from functools import partial
    from rpy2.rinterface import initr
    rinterface.set_initoptions(("--max-ppsize=100000"))
    initr()
    client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
    lview = client.load_balanced_view() # default load-balanced view
    lview.block = True
    # func = lambda args: train_hmm_n_times(file_id=args[0], nstates=args[1], trials=args[2], iter=args[3])
    # trials = 4
    client[:].push(dict(train_hmm_once=train_hmm_once))
    # args = [(file_id, nstates, trials, 1000) for nstates in range(5,10)]
    # results = lview.map(func, args)# hmm, d, results = train_hmm_n_times(file_id, nstates, trials=20, iter=1000)
    # pool.join()
    results = {}
    for i in range(3):
        results[i] = train_hmm_n_times(file_id, nstates=nstates, trials=trials, iter=iter, phase=i)
    return results

def analyze_log_file_in_phases_by_condition(file_id, nstates, trials, iter, units=constants.XY, parallel=True,
                                            prefix="logs/", skip_phases=[]):
    import gc
    print_n_flush( "Starting phase by phase analysis, controlled for conditions (units: %s)..." % units)
#     d = pd.read_csv("/shared/AudioData/ThereminData/surfacedata.csv", na_values=["NaN"])
    global id_to_log
    id_to_log = lambda x: "%s/%s.exp.log" % (prefix, x)
    filename_log = id_to_log(file_id)
    cond = file_id.split('.')[-1]
    print_n_flush( "Condition", cond)
    responses, tests, responses_t, tests_t, images = toCSV(filename_log)

    #     client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
#     lview = client.load_balanced_view() # default load-balanced view
#     lview.block = True
    # func = lambda args: train_hmm_n_times(file_id=args[0], nstates=args[1], trials=args[2], iter=args[3])
    # trials = 4
#     client[:].push(dict(train_hmm_once=train_hmm_once))
    # args = [(file_id, nstates, trials, 1000) for nstates in range(5,10)]
    # results = lview.map(func, args)# hmm, d, results = train_hmm_n_times(file_id, nstates, trials=20, iter=1000)
    # pool.join()
    results = {}
    for i in range(3):
        if str(i) in skip_phases:
            print_n_flush("Skipping phase#%d" % i)
            continue
        print_n_flush("Doing phase#%d" % i)
        results[i] = train_hmm_n_times(file_id, nstates=nstates, trials=trials, iter=iter, phase=i, cond=cond,
                                       units=units, parallel=parallel, pickle=True)
        gc.collect()
    return results

# <codecell>

def pull_hmm_paths(d):
    globalenv['d'] = d
    r('library("RHmm")')
    r('path = list()')
    r('for(trajectory in d){ path = c(path, viterbi(hmm, trajectory))}')
    path = globalenv['path']
    return path
# from glob import glob
# files = glob("logs/*.*.exp.log")
# files = [file for file in files if file[:-8].split(".")[-1] in ('master','1','1r','2','2r')]
# print_n_flush( files, len(files))
# for f in files:
#     analyze_log_file_in_phases_by_condition(f[5:-8], 5, 1, 100)

# <codecell>

# line = 'hmms = analyze_log_file_in_phases_by_condition("1320116514.2", nstates=range(2,30), trials=5, iter=100, parallel=True, units=Constants.XY)'
# %lprun -f analyze_log_file_in_phases_by_condition analyze_log_file_in_phases_by_condition("1320116514.2", nstates=range(2,30), trials=5, iter=100, parallel=True, units=Constants.XY)
# hmms_f = analyze_log_file_in_phases_by_condition("1320116514.2", nstates=10, trials=50, iter=100, parallel=False, units=Constants.AMP_AND_FREQ)
# hmms_m = analyze_log_file_in_phases_by_condition("1320116514.2", nstates=10, trials=50, iter=100, parallel=False, units=Constants.AMP_AND_MEL)

# <codecell>

# %lprun?
# %matplotlib qt
# import ghmm
# hmm = hmms[1]
# print hmm.multivariate
# # print hmm
# # print hmm.univariate
# # print hmm
# # plot_hmm(means_ = hmm.means, transmat = hmm.transmat, covars=hmm.variances, initProbs=hmm.initProb)

# <codecell>

# %matplotlib inline
# from GHmmWrapper import nest
# from matplotlib import colors
# from matplotlib.pyplot import figure
# # clr = [colorConverter.to_rgb(c) for c in colors.cnames]
# def plot_hmm_obj(hmm, 
#                  clr=[colorConverter.to_rgb(c) for c in colors.cnames],
#                  transition_arrows=True):
#     plot_hmm(means_ = hmm.means, transmat = hmm.transmat, 
#              covars=hmm.variances, initProbs=hmm.initProb,
#              clr=clr, transition_arrows=transition_arrows)
# # plot_hmm_obj(hmms[1], clr=clr)
# def plot_hmm_and_trajectories(hmm, 
#                               clr=[colorConverter.to_rgb(c) for c in colors.cnames],
#                               transition_arrows=True,
#                               separate_figures=False,
#                               traj_list=[],
#                               units=Constants.XY):
#     if not separate_figures:
#         plot_hmm_obj(hmm, clr=clr, transition_arrows=transition_arrows)
        
#     for i,dd in enumerate(hmm.training_data):
#         if traj_list != []:
#             if i not in traj_list:
#                 continue
#         if separate_figures:
#             figure()
#             plot_hmm_obj(hmm, clr=clr, transition_arrows=transition_arrows)
#         d = nest(dd)
#         path = hmm.viterbi_path(dd)[0]
# #         print path
#         # xmin, xmax, ymin, ymax = find_bounding_box([d])
#         # print list(d)
#         C=[clr[i] for i in path]
# #         print path
# #         print C
#         plot_quiver2d(d, C=C,alpha=.7)
# #         break
# # plot_hmm_and_trajectories(hmm, 
# #                           transition_arrows=False, 
# #                           separate_figures=True,
# #                           traj_list=[])
# from matplotlib.pyplot import figure;figure()
# plot_hmm_and_trajectories(hmm, 
#                           transition_arrows=False, 
#                           separate_figures=False,
#                           traj_list=[])
# #     break
# #     traj = to_trajectory_object_raw(d, *find_bounding_box_raw([d]))
# #     plot_hmm_path([traj],[[1 for i in hmms[1].training_data]], axis=gca())

# <codecell>

# %Rpull hmm
# hmm[1][0]

# <codecell>

# import rpy2.robjects.numpy2ri
# from rpy2.robjects import r, globalenv

# results = analyze_log_file(file_id, nstates = range(10,22), trials = 20, iter = 1000)
# hmm, d = pick_hmm_by_bic(results, responses, responses_t, plot=False)

# path = pull_hmm_paths(d)

# paths = [numpy.asarray(path[i], dtype=int) for i in range(0, len(path), 3)]
# tr, trajectories = responses_to_traj_objs(responses,responses_t)

def draw():
    means = hmm.means
    transmat = hmm.transmat
    initProb = hmm.initProb
    covar = hmm.variances
    nstates = hmm.nstates
    x = zip(*means)[0]
    y = zip(*means)[1]

    legends = []
    items = []
    ax = None

    plot_hmm(numpy.asarray(means), 
            numpy.asarray(transmat),
            initProbs=numpy.asarray(initProb),
            covars=covar, axes=ax)

    plot_hmm_path(paths=paths, 
                  trajectory_objs=tr, 
                  legends=legends, 
                  items=items)
    plt.draw()
    plt.show()

# <codecell>

# responses, test_results, responses_p, test_p, images = fromFile(id_to_log(file_id))
# data = responses["127.0.0.1"][str(1)].values()
# data = [[frame.get_stabilized_position()[:2] for frame in response] for response in data]



# <codecell>

# from itertools import product
# print pd.DataFrame(data[0])[:3]
# # print [[x for x,y in seq] for seq in data][:3]
# print get_range_of_multiple_traj(data)
# # min([el[0] for el in seq for seq in data])
# # data[0]

# <codecell>

# from pickle import dumps, dump
# from ExperimentalData import reconstruct_hmm, reduce_hmm
# import copy_reg
# copy_reg.pickle(HMM, reduce_hmm, reconstruct_hmm)
# print dumps(hmm)
# results = unpickle_results("logs/1230115514.master.exp.log", phase=2, units=Constants.AMP_AND_FREQ)
# print results.hmms[0]

# <codecell>

# results.hmms[0].entropy_rate()

# <codecell>

def stationary(transmat):
		from rpy2.robjects import r, globalenv
        import pandas as pd
		import pandas.rpy.common as com

        r("library('DTMCPack')")
		globalenv['transmat'] = com.convert_to_r_dataframe(pd.DataFrame(transmat))
		stationary_dist = r("statdistr(transmat)")
		# long_as = lambda x: range(len(x)) 
# 		rate = 0
# 		for s1, s2 in product(range(len(self.means)),range(len(self.means))):
# 			p = self.transmat[s1][s2]
# 			rate -= stationary_dist[s1] * xlogy(p,p)
		return stationary_dist
# print stationary(results.hmms[1].transmat)

# <codecell>

# import numpy as np
# from numpy.linalg import matrix_power
# from scipy.linalg import eig
# m = np.asmatrix(results.hmms[1].transmat)
# print m
# final = None
# for i in range(50000):
# #     if not i % 1000:
#         m_ = matrix_power(m, i)
# #         print m_
#         w, vl, vr = eig(m_, left=True)
#         for i, w_ in enumerate(w):
#             if w_ != 1.0:
# #             if not (0.999 <= w_ <= 1.001):
#                 continue
# #             print "Eig-vec:", vl_
# #             if not (0.95 <= sum(vl_) <= 1.05):
# #                 continue
# #             skip = False
# #             for v in vl_:
# #                 if v < 0:
# #                     skip = True
# #                     break
# #             if skip:
# #                 break
# #             print "Raw:", vl[:,i].T
#             normalized = vl[:,i].T / sum(vl[:,i].T)
# #             print "Normalized:", normalized
# #             print "Sum:", sum(normalized)
#             final = normalized
# print "Final:", normalized

