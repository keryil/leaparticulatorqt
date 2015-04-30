# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Import data and output to file

# <codecell>

import pandas as pd
import jsonpickle
import numpy as np

from leaparticulator.data.functions import toCSV
from leaparticulator.data.hmm import HMM


colors = [(x / 10., y / 20., z / 40.) for x, y, z in zip(range(10), range(10), range(10))]
colors.extend([(x / 40., y / 20., z / 10.) for x, y, z in zip(range(1, 10), range(1, 10), range(1, 10))])
colors.extend([(x / 40., y / 10., z / 20.) for x, y, z in zip(range(1, 10), range(1, 10), range(1, 10))])
colors.extend(['red', 'green', 'yellow', 'magenta', 'orange', 'black', 'cyan', 'white'])

pd.set_option("display.max_columns", None)
file_id = "1230105514.master"
id_to_log = lambda x: "logs/%s.exp.log" % x
# filename_log = id_to_log(file_id)
# responses, tests, responses_t, tests_t, images = toCSV(filename_log)

# <codecell>

# quiver_annotations = []
from trajectory import Trajectory


def plot_quiver2d(data, alpha=.75, C=[], path=None, *args, **kwargs):
    global quiver_annotations

    X, Y = zip(*tuple(data))
    U = [x1 - x0 for x0, x1 in zip(X[:-1], X[1:])]
    V = [y1 - y0 for y0, y1 in zip(Y[:-1], Y[1:])]
    if C == []:
        color_delta = 1. / (len(X) - 1)
        C = [(color_delta * i, color_delta * i, color_delta * i) for i in range(len(X) - 1)]
    # print_n_flush(_n_flush( C))
    X, Y = X[:-1], Y[:-1]
    # print_n_flush(_n_flush( X, Y, U, V))
    patches = quiver(X, Y, U, V, C, *args, scale_units='xy', angles='xy', scale=1, width=0.005, alpha=alpha, **kwargs)
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
                   ndim=2, dim_size=(xmax - xmin, ymax - ymin), prob_c=1)
    return t


def to_trajectory_file(trajectories, filename):
    xmin, xmax, ymin, ymax = find_bounding_box(trajectories)
    start = 0
    end = 1
    import os

    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        print_n_flush(_n_flush((xmin, xmax, ymin, ymax, start, end)))
        f.write("%d %d %d %d %d %d\n" % (xmin, xmax, ymin, ymax, start, end))
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
                # if image == u'./img/meanings/5_1.png':
                counter += 1
                trajectory = r[phase][image]
                trajectories.append(trajectory)
                data = []
                for frame in trajectory[:-1]:
                    x, y, z = frame.get_stabilized_position()
                    data.append((x, y))
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
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

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
    print_n_flush(event)
    print_n_flush(annotations)
    if event.artist in annotations:
        on_pick_annotation(event)
    elif event.artist in means:
        on_pick_means(event)
    draw()
    time.sleep(1)


def on_pick_trajectory_event(event):
    pass


def on_pick_annotation(event):
    print_n_flush("Annotation:", event.artist)
    event.artist.set_visible(False)


def on_pick_means(event):
    print_n_flush("Mean:", means.index(event.artist))
    annotations[means.index(event.artist)].set_visible(True)
    print_n_flush(annotations[means.index(event.artist)])


# colors = ['red','green','yellow', 'magenta', 'orange', 'black', 'cyan', 'white']
def plot_hmm(means_, transmat, covars, initProbs, axes=None):
    if axes != None:
        axes(axes)
        # f, axes = subplots(2)#,sharex=True, sharey=True)
    #     sca(axes[0])
    global annotations
    annotations = []
    global means
    means = []
    color_map = colors  # [colorConverter.to_rgb(colors[i]) for i in range(len(means_))]
    for i, mean in enumerate(means_):
        # print_n_flush( "MEAN:", tuple(mean))
        means.append(scatter(*tuple(mean), color=colorConverter.to_rgb(colors[i]), picker=10, label="State%i" % i))
        annotate(s="%d" % i, xy=mean, xytext=(-10, -10), xycoords="data", textcoords="offset points",
                 alpha=1, bbox=dict(boxstyle='round,pad=0.2', fc=colorConverter.to_rgb(colors[i]), alpha=0.3))
        #         gca().add_patch(Ellipse(xy = means_[i], width = np.diag(covars[i])[0], height = np.diag(covars[i])[1],
        #                         alpha=.15, color=colorConverter.to_rgb(colors[i])))
        plot_cov_ellipse(covars[i], mean, alpha=.15, color=colorConverter.to_rgb(colors[i]))
        x0, y0 = mean
        prob_string = "P(t0)=%f" % initProbs[i]
        for j, p in enumerate(transmat[i]):
            xdif = 10
            ydif = 5
            s = "P(%d->%d)=%f" % (i, j, p)
            #             print_n_flush( "State%d: %s" % (i, s))
            prob_string = "%s\n%s" % (prob_string, s)
            if i != j:
                x1, y1 = means_[j]
                # if transmat[i][j] is too low, we get an underflow here
                #                 q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = 10000 * (transmat[i][j]**2),
                alpha = 10 ** -300
                if p > 10 ** -100:
                    alpha = (100 * p) ** 2
                q = quiver([x0], [y0], [x1 - x0], [y1 - y0], alpha=1 / log(alpha),
                           scale_units='xy', angles='xy', scale=1, width=0.005, label="P(%d->%d)=%f" % (i, j, p))
        legend()

        annotations.append(annotate(s=prob_string, xy=mean, xytext=(0, 10), xycoords="data", textcoords="offset points",
                                    alpha=1,
                                    bbox=dict(boxstyle='round,pad=0.2', fc=colorConverter.to_rgb(colors[i]), alpha=0.3),
                                    picker=True,
                                    visible=False))

        print_n_flush("State%i is %s" % (i, colors[i]))
    cid = gcf().canvas.mpl_connect('pick_event', on_pick)


# <codecell>

def plot_hmm_path(trajectory_objs, paths, legends=[], items=[]):
    global colors
    print_n_flush("Colors are:", colors)
    for i, (trajectory, p) in enumerate(zip(trajectory_objs, paths)):
        print_n_flush("Path:", p)
        tr_colors = [colors[int(state)] for state in p]
        t = trajectory.plot2d(color=tr_colors)
        # t = plot_quiver2d(trajectory, color=tr_colors, path=p)
        too_high = [tt for tt in trajectory if tt[1] > 400]
        #         print_n_flush( "Too high", too_high)
        legends.append("Trajectory%i" % i)
        #     items.append(p)
        items.append(t)
    # gca().legend()



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

# from IPython.parallel import Client
# from functools import partial
# from rpy2.rinterface import initr
# initr()
# client = Client(profile="default")
# # client[:].push(dict(initr=initr))
# # client[:].apply_sync(lambda: initr())
# lview = client.load_balanced_view() # default load-balanced view
# lview.block = True
from leaparticulator import constants


def train_hmm_once(file_id, nstates, iter=1000, phase=2, cond=None, units=constants.XY):
    """
    Trains a Gaussian HMM using the data from an experimental log file, 
    using the specified number of states. When condition is unspecified,
    we assume condition 1. 
    """
    from IPython.parallel import CompositeError
    from rpy2.rinterface import RRuntimeError

    cond_text = None
    phase = int(phase)
    x, y = None, None
    if units == constants.AMP_AND_FREQ:
        x, y = "frequency", "amplitude"
    elif units == constants.AMP_AND_MEL:
        x, y = "mel", "amplitude"
    elif units == constants.XY:
        x, y = "x", "y"
    print_n_flush("Units: %s, x: %s, y: %s" % (units, x, y))
    cond_text = "c('%s','%s')" % (x, y)
    if cond == "master":
        cond = "1"
    if cond is None:
        pass
    elif phase == 0:
        if "r" not in cond:
            cond_text = "c('%s')" % x
        else:
            cond_text = "c('%s')" % y
    elif phase == 1:
        if "2" in cond:
            cond_text = cond_text
        elif "r" in cond:
            cond_text = "c('%s')" % y
        else:
            cond_text = "c('%s')" % x
    elif phase == 2:
        if "1" in cond:
            cond_text == cond_text
        elif "r" in cond:
            cond_text = "c('%s')" % y
        else:
            cond_text = "c('%s')" % x
    else:
        raise Exception("Invalid phase %s" % phase)

        # from rpy2.rinterface import initr
    #     initr()
    #     %load_ext rpy2.ipython
    from rpy2.robjects import r, globalenv
    # initr()
    r("rm(list = setdiff(ls(), lsf.str()))")
    r("source(\"~/Dropbox/ABACUS/Workspace/LeapArticulator/SampleHMM.R\")")
    #     %R source("~/Dropbox/ABACUS/Workspace/LeapArticulator/SampleHMM.R")
    #     r('file_id = %s' % file_id)
    #     r('nstates = %s' % nstates)
    #     robjects.globalenv['file_id'] = file_id
    #     robjects.globalenv['nstates'] = nstates
    #     %Rpush file_id
    #     %Rpush nstates
    #     %R list[hmm, d] = fitHMM(file_id, nstates, iter=1000)
    #     %Rpull d
    #     %Rpull hmm
    success = False
    attempts = 1
    while not success:
        try:
            command = "list[hmm, d] <- fitHMMtoPhase(file_id='%s', nStates=%s, iter=%d, phase=%d, take_vars=%s);" % (
            file_id, nstates, iter,
            phase, cond_text)
            print_n_flush("R command (attempt %d): %s" % (attempts, command))
            attempts += 1
            r(command)
            hmm, d = globalenv['hmm'], globalenv['d']
            converged = False
            if hmm is not None:
                converged = r("hmm$convergence")
                #     print_n_flush( "Finished a %s state run" % nstates)
                #     print_n_flush( "BIC:", hmm.rx("BIC")[0][0])
                #     print_n_flush( "AIC:", hmm.rx("AIC")[0][0])
                #     print_n_flush( "Returning:", (hmm,d))
            if not converged:
                print_n_flush(
                    "Returning a null hmm for nstates=%s, cond:ph=%s:%s, units=%s" % (nstates, cond, phase, units))
                hmm = None
            return hmm, d
        except RRuntimeError, e:
            print_n_flush("RRuntimeError: %s" % e)
            continue
            #                 write("RRuntimeError: %s" % e)
        except CompositeError, e:
            print_n_flush("Composite error ******")
            print_n_flush(e)
            continue
            #                 write("Composite error ******")
            #                 write(e)
            #             e.raise_exception()
        except Exception, e:
            print_n_flush("Error during analysis: ")
            print_n_flush(e, e.args)
            print_n_flush(traceback.format_exc())
            continue
            #                 write("Error during analysis: ")
            #                 write( e, e.args)
            #                 write( traceback.format_exc())
        except:
            print_n_flush("Some other exception that is not an exception.")
            continue
            #                 write("Some other exception that is not an exception.")
        success = True


def train_hmm_n_times(file_id, nstates, trials=20, iter=1000, pickle=True,
                      phase=2, cond=None, units=constants.XY, parallel=True):
    """
    Trains multiple HMM's (as many as trials parameter per nstate) and chooses the one with the 
    lowest BIC, so as to avoid local optima. units parameter can be "xy", "amp_and_freq", or
    "amp_and_mel", which specifies the kind of data to fit the HMM to. 
    """

    def pick_lowest_bic(models):
        hmm, d, bic = None, None, 9999999999
        for a in models:
            # print_n_flush( list(a))
            hmmm, dd = a
            if hmmm is None:
                continue
            # use this for ipcluster cases
            bbic = hmmm[2][0]
            # use this for non-cluster cases
            # bbic = hmmm.rx("BIC")[0][0]
            if bbic < bic:
                bic = bbic
                hmm = hmmm
                #                 print_n_flush( "Type:", type(dd))
                d = [np.array(i) for i in dd]
                #                 np.asarray(d)
                d = np.asarray(d)
        if hmm is None:
            return (None, d)
        Hmm = HMM(hmm, training_data=d)
        # print_n_flush( "New hmm and data (%s)" % d)
        #         Hmm.from_R(hmm)
        return (Hmm, d)

    from IPython.parallel import Client

    client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
    lview = client.load_balanced_view()  # default load-balanced view

    lview.block = True
    hmm, d, bic = None, None, None

    def do_train_svp(args):
        return train_hmm_once(file_id=args[0], nstates=args[1], iter=args[2],
                              phase=args[3], cond=args[4], units=args[5])

    my_map = lview.map
    if not parallel:
        my_map = map
    results = []
    args = []
    lview.block = False
    if len(nstates) != 1:
        for nstate in nstates[:-1]:
            args = [(file_id, nstate, iter, phase, cond, units)] * trials
            rr = my_map(do_train_svp, args)
            results.append(rr)
            # results.append(map(func, args))
            print_n_flush("Submitted the %d state models (Are some valid? %s)" % (nstate, any([rrr[0] for rrr in rr])))
            # print_n_flush( rr )
    lview.block = True
    args = [(file_id, nstates[-1], iter, phase, cond, units)] * trials
    results.append(my_map(do_train_svp, args))
    # results.append(map(func,args))

    to_return = []
    for nstates_r in results:
        to_return.append(pick_lowest_bic(nstates_r))
    if pickle is True:
        pickle_results(to_return, nstates, trials, iter, id_to_log(file_id), phase, units=units)
    return to_return


def pickle_results(results, nstates, trials, iter, filename_log, phase=None, units=constants.XY):
    hmms, ds = zip(*results)
    print_n_flush(hmms)
    # results_pickled = jsonpickle.encode((hmms, ds, nstates, trials, iter))
    extension = ".hmms"
    if (phase is not None):
        extension = ".phase%d.%s.hmms" % (phase, units)
    print_n_flush("Writing results to %s" % (filename_log + extension))
    with open(filename_log + extension, "w") as f:
        for i, item in zip(("hmms", "ds", "nstates", "trials", "iter"), (hmms, ds, nstates, trials, iter)):
            print_n_flush(i)
            #             if i == "hmms":
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
    from leaparticulator.data import functions
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
        hmms = jsonpickle.decode(f.readline().rstrip())
        for hmm in hmms:
            if "entropy_rate" not in dir(hmm) and hmm is not None:
                hmm.entropy_rate = HMM.entropy_rate
        ds = jsonpickle.decode(f.readline().rstrip())
        nstates = jsonpickle.decode(f.readline().rstrip())
        trials = jsonpickle.decode(f.readline().rstrip())
        iter = jsonpickle.decode(f.readline().rstrip())
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
    tr = [to_trajectory_object(trajectory, step_size=300, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) for trajectory in
          all_trajectories]
    return tr, trajectories


def pick_hmm_by_bic(hmms, responses, responses_t, plot=True):
    responses_to_traj_objs(responses, responses_t)
    states, bics, aics = [], [], []
    best = 0
    tr, trajectories = responses_to_traj_objs(responses, responses_t)
    for i, (hmm, d) in enumerate(results):
        if hmm is not None:
            # nstates = hmm[0][2][1][0]
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
            # print_n_flush( states)
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
    # print_n_flush( best)
    #     print_n_flush( aics)
    return hmm, d


# <codecell>

def analyze_log_file(file_id, nstates, trials, iter):
    id_to_log = lambda x: "logs/%s.exp.log" % x
    filename_log = id_to_log(file_id)
    responses, tests, responses_t, tests_t, images = toCSV(filename_log)

    from IPython.parallel import Client
    # from functools import partial
    from rpy2.rinterface import initr

    try:
        rinterface.set_initoptions(("--max-ppsize=500000"))
    except RuntimeError, e:
        print_n_flush("Runtime error, probably redundant call to set_initoptions()")
        print_n_flush(e)
    initr()
    client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
    lview = client.load_balanced_view()  # default load-balanced view
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
    print_n_flush("Starting phase by phase analysis...")
    id_to_log = lambda x: "logs/%s.exp.log" % x
    filename_log = id_to_log(file_id)
    responses, tests, responses_t, tests_t, images = toCSV(filename_log)
    from IPython.parallel import Client
    # from functools import partial
    from rpy2.rinterface import initr

    rinterface.set_initoptions(("--max-ppsize=100000"))
    initr()
    client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
    lview = client.load_balanced_view()  # default load-balanced view
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


def analyze_log_file_in_phases_by_condition(file_id, nstates, trials, iter, units=constants.XY, parallel=True):
    print_n_flush("Starting phase by phase analysis, controlled for conditions (units: %s)..." % units)
    d = pd.read_csv("/shared/AudioData/ThereminData/surfacedata.csv", na_values=["NaN"])

    id_to_log = lambda x: "logs/%s.exp.log" % x
    filename_log = id_to_log(file_id)
    cond = file_id.split('.')[-1]
    print_n_flush("Condition", cond)
    responses, tests, responses_t, tests_t, images = toCSV(filename_log)
    from IPython.parallel import Client
    from rpy2 import rinterface

    try:
        rinterface.set_initoptions(("--max-ppsize=500000", "--vanilla"))
    except RuntimeError, e:
        print_n_flush("Runtime error, probably redundant call to set_initoptions()")
        print_n_flush(e)
    # from functools import partial
    from rpy2.rinterface import initr

    initr()
    client = Client(profile="default")
    # client[:].push(dict(initr=initr))
    # client[:].apply_sync(lambda: initr())
    lview = client.load_balanced_view()  # default load-balanced view
    lview.block = True
    # func = lambda args: train_hmm_n_times(file_id=args[0], nstates=args[1], trials=args[2], iter=args[3])
    # trials = 4
    client[:].push(dict(train_hmm_once=train_hmm_once))
    # args = [(file_id, nstates, trials, 1000) for nstates in range(5,10)]
    # results = lview.map(func, args)# hmm, d, results = train_hmm_n_times(file_id, nstates, trials=20, iter=1000)
    # pool.join()
    results = {}
    for i in range(3):
        results[i] = train_hmm_n_times(file_id, nstates=nstates, trials=trials, iter=iter, phase=i, cond=cond,
                                       units=units, parallel=parallel)
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
# analyze_log_file_in_phases_by_condition(f[5:-8], 5, 1, 100)

# <codecell>


# <codecell>

##%load_ext rpy2.ipython

# <codecell>


# getwd()
# source('SampleHMM.R')
# list[hmm, d] <- fitHMM(file_id = '123R0126514.1r', nStates = 6)

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

# <rawcell>


# <rawcell>


