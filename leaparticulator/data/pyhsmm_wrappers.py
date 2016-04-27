import copy

import matplotlib.pyplot as plt
import pyhsmm
from pyhsmm.util.text import progprint_xrange

from leaparticulator.constants import kelly_colors
from leaparticulator.data.hmm import *


def plot_viterbi(hmm, s, ax=None, plot_slice=slice(None), update=False, draw=True, state_colors=None):
    s = hmm.states_list[s] if isinstance(s, int) else s
    ax = ax if ax else plt.gca()
    state_colors = hmm._get_colors(scalars=True) if state_colors is None else state_colors

    hmm._plot_stateseq_pcolor(s, ax, state_colors, plot_slice, update)
    data_values_artist = hmm._plot_stateseq_data_values(s, ax, state_colors, plot_slice, update)

    if draw: plt.draw()

    return [data_values_artist]


def plot_hdp_hmm(hmm, axes=None, clr=kelly_colors, transition_arrows=True,
                 prob_lists=True, legend=True, verbose=False, *args, **kwargs):
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch, ArrowStyle

    univariate = hmm.obs_distns[0].mu.shape[0] == 1

    if univariate:
        print "Univariate HMM detected (%d states)." % len(hmm.means)
        means_ = [(0, d.mu) for d in hmm.obs_distns]
    else:
        means_ = [d.mu for d in hmm.obs_distns]
    covars = [d.sigma for d in hmm.obs_distns]

    if axes is None:
        axes = plt.gca()

    hmm.plot_annotations = []
    hmm.plot_means = []
    hmm.plot_covars = []
    # hmm.cid = None

    arrows = []
    colors = clr
    transmat = hmm.trans_distn.trans_matrix
    max_prob = np.max(transmat)
    # for i, row in enumerate(transmat):
    #     for j, p in enumerate(row):
    #         # ignore self-transitions
    #         if i != j:
    #             max_prob = max(max_prob, p)

    #     max_prob = max(transmat.flatten())
    for i, mean in enumerate(means_):
        if hmm.state_usages[i] == 0:
            if verbose:
                print "Skipping unused state %i" % (i)
            continue
        color = colors[i % len(colors)]
        if verbose:
            print  "MEAN:", tuple(mean)
        hmm.plot_means.append(axes.scatter(*tuple(mean), color=color, picker=10, label="State%i" % i))
        axes.annotate(s="%d" % i, xy=mean, xytext=(-10, -10), xycoords="data", textcoords="offset points",
                      alpha=1, bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.3))
        if verbose:
            print  "COVARS: %s" % covars[i]
        if not univariate:
            if verbose:
                print "Drawing ellipse..."
            hmm.plot_covars.append(plot_cov_ellipse(covars[i], mean, alpha=.30, color=color, ax=axes))
        else:
            hmm.plot_covars.append(
                axes.axhspan(mean[1] - np.sqrt(covars[i]), mean[1] + np.sqrt(covars[i]), color=color, alpha=.30))
        x0, y0 = mean
        prob_string = "$\pi = %f$" % hmm.init_state_distn.pi_0[i]
        for j, p in enumerate(transmat[i]):
            xdif = 10
            ydif = 5
            s = "$A(S_%d) = %f$" % (j, p)
            if p < 10 ** - 20:
                continue
                #             print_n_flush( "State%d: %s" % (i, s))
            prob_string = "%s\n%s" % (prob_string, s)
            if transition_arrows:
                if i != j:
                    x1, y1 = means_[j]
                    # if transmat[i][j] is too low, we get an underflow here
                    #                 q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = 10000 * (transmat[i][j]**2),
                    alpha = 0
                    if p > 10 ** -300:
                        alpha = (p * 100000) / (max_prob * 100000)
                    alpha = min(1., (np.exp(2 * alpha / (len(means_) * .5))) - 1)

                    #                     alpha_in = 0
                    #                     if transmat[j][i] > 10 ** -300:
                    #                         alpha_in = (transmat[j][i]*100000) / (max_prob * 100000)
                    #                     alpha_in = min(1.,(exp(alpha_in / (len(means)*.5))) - 1)

                    #                     print alpha, old_alpha, p, max_prob
                    #                     alpha = max(0, 1. / log(alpha))
                    width = .55
                    color = "red"
                    if x1 > x0:
                        color = "green"
                        #                     if j > i:
                    c_arrows = FancyArrowPatch(
                        (x0, y0),
                        (x1, y1),
                        connectionstyle='arc3, rad=-.25',
                        mutation_scale=10,
                        # red is forward, green is backward prob
                        color=color,
                        alpha=alpha,
                        linewidth=width,
                        arrowstyle=ArrowStyle.Fancy(head_length=width * 4,
                                                    head_width=width * 2.5,
                                                    ))
                    #                     c_arrows = FancyArrow(x0, y0, x1-x0, y1-y0, alpha=alpha, color="black",
                    #                                          width=width, head_width=width * 2.5, head_length=width * 4.,
                    #                                          overhang=1.)
                    #                                             connectionstyle="angle3,angleA=0,angleB=-90")
                    axes.add_patch(c_arrows)
                    arrows.append(c_arrows)
                    #                     q = axes.quiver([x0], [y0], [x1-x0], [y1-y0], alpha = alpha,
                    #                            scale_units='xy',angles='xy', scale=1, width=0.005,
                    #                             label="P(%d->%d)=%f" % (i,j,p))
                    #         legend()
        if prob_lists:
            hmm.plot_annotations.append(
                plt.annotate(s=prob_string, xy=mean, xytext=(0, 10), xycoords="data", textcoords="offset points",
                             alpha=1, bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.3), picker=True,
                             visible=True))

    # if hmm.cid:
    #     plt.gcf().canvas.mpl_disconnect(hmm.cid)
    # hmm.cid = plt.gcf().canvas.mpl_connect('pick_event', hmm.on_pick)
    if legend:
        axes.legend()
    hmm.plot_arrows = arrows
    if verbose:
        print "Returning from plot_hmm"
    return hmm.plot_annotations, hmm.plot_means, arrows


def train_hdp_hmm(data_sets, nmax, niter, nsamples):
    """

    :param data_sets: a list of ndarrays.
    :param nmax:
    :param niter:
    :param nsamples:
    :return:
    """
    interval = (niter - 1) / (nsamples)
    obs_dim = data_sets[0].shape[-1]
    obs_hypparams = {'mu_0': np.zeros(obs_dim),
                     'sigma_0': np.eye(obs_dim),
                     'kappa_0': 0.25,
                     'nu_0': obs_dim + 2}

    ### HDP-HMM without the sticky bias

    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(nmax)]
    posteriormodel = pyhsmm.models.WeakLimitHDPHMM(alpha=6., gamma=6., init_state_concentration=1.,
                                                   obs_distns=obs_distns)
    for d in data_sets:
        posteriormodel.add_data(d)

    models = []
    last = None
    for idx in progprint_xrange(niter):
        posteriormodel.resample_model()
        # print idx
        if idx % interval == 0 and idx != 0:
            models.append(copy.deepcopy(posteriormodel))
        last = posteriormodel
    models[nsamples - 1] = posteriormodel
    return models


def train_sticky_hdp_hmm(data_sets, nmax, niter, nsamples):
    """

    :param data_sets: a list of ndarrays.
    :param nmax:
    :param niter:
    :param nsamples:
    :return:
    """
    interval = niter / nsamples
    obs_dim = data_sets[0].shape[-1]
    obs_hypparams = {'mu_0': np.zeros(obs_dim),
                     'sigma_0': np.eye(obs_dim),
                     'kappa_0': 0.25,
                     'nu_0': obs_dim + 2}

    ### HDP-HMM without the sticky bias

    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(nmax)]
    posteriormodel = pyhsmm.models.WeakLimitStickyHDPHMM(kappa=50., alpha=6., gamma=6., init_state_concentration=1.,
                                                         obs_distns=obs_distns)
    for d in data_sets:
        posteriormodel.add_data(d)

    models = []
    last = None
    for idx in progprint_xrange(niter):
        posteriormodel.resample_model()
        if idx % interval == 0:
            models.append(copy.deepcopy(posteriormodel))
        last = posteriormodel
    models[nsamples - 1] = posteriormodel
    return models
