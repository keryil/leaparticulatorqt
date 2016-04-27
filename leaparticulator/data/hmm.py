import ghmm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from leaparticulator import constants

__author__ = 'Kerem'


def check_two_nonzeroes(vector):
    nonzero = 0
    for v in vector:
        if v > 0:
            nonzero += 1
            if nonzero > 1:
                return True
    return False


def flatten_to_emission(arr, domain=ghmm.Float()):
    arr = np.array(arr).flatten().tolist()
    return ghmm.EmissionSequence(domain, arr)


def to_emission_set(arrs, domain=ghmm.Float()):
    set = []
    for arr in arrs:
        set.append(flatten_to_emission(arr, domain))
    return ghmm.SequenceSet(set)


def nest(arr, domain=ghmm.Float(), tolist=True):
    if isinstance(arr, ghmm.SequenceSet):
        to_return = []
        for a in arr:
            assert len(a) % 2 == 0
            a = np.array(a).reshape((len(a) / 2, 2))
            if tolist:
                a = a.tolist()
            to_return.append(a)
        return to_return
    else:
        assert len(arr) % 2 == 0
        arr = np.array(arr).reshape((len(arr) / 2, 2))
        if tolist:
            arr = arr.tolist()
        return arr


# #####
# PLOTTING
# #####
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
    from matplotlib.patches import Ellipse
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(np.asarray(cov).reshape((2, 2)))
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, ec="black", **kwargs)

    ax.add_artist(ellip)
    return ellip

class HMM(object):
    """
    This class encapsulates R (and other) HMM objects. The purpose is to have a single
    pure python object that can represent HMMs accross platforms.
    """

    # means = []
    # nstates = []
    # variances = []
    # initProb = []
    # transmat = []
    # loglikelihood = None
    # bic = None
    # aic = None
    # training_data = None
    # mixtures = 1
    # hmm_object = None
    # hmm_type = None

    def __init__(self, hmm_obj=None, training_data=None,
                 matrices=None, hmm_type="ghmm",
                 nstates=None):
        assert hmm_type is not None
        self.means = []
        self.nstates = nstates
        self.variances = []
        self.initProb = []
        self.transmat = []
        self.loglikelihood = None
        self.bic = None
        self.aic = None
        self.training_data = None
        self.mixtures = 1
        self.hmm_object = None
        self.plot_means = []
        self.plot_annotations = []
        self.plot_arrows = []
        self.cid = None

        self.hmm_type = hmm_type
        if hmm_type == "R":
            self.from_R(hmm_obj, training_data)
        elif hmm_type == "ghmm":
            if hmm_obj is not None:
                self.from_ghmm(hmm_obj, training_data)
            else:
                if self.nstates and (not matrices):
                    self._induce_from_data(training_data)
                else:
                    self.from_matrix(matrices, training_data)

    def _induce_from_data(self, training_data):
        """
        This method assigns observed (k)means to states,
        and the observed variances as covariance matrices.
        Expects self.nstates to be defined when it is called.
        :return:
        """
        assert self.nstates
        multivariate = len(training_data[0]) > 1
        obs = [flatten_to_emission(d) for d in training_data]
        obs = ghmm.SequenceSet(domain, obs)

        means = list(KMeans(n_clusters=self.nstates).fit(training_data).cluster_centers_)
        variance = np.cov(list(obs[rand_obs])[::2], list(obs[rand_obs])[1::2]).flatten()
        domain = ghmm.Float()

        self.hmm_object.baumWelch(obs)
        self.hmm_object.obs = obs

                # def __reduce__(self):
                # return (reconstruct_hmm,

    # (self.hmm_object.asMatrices(), nest(self.training_data)))


    # def reduce(self):
    # 	return self.__reduce__()

    # from now onwards, it's ghmm specific

    def nparams(self):
        """
        Returns the number of (free) parameters in an HMM. I copied this from the ghmm
        C++ code where it exists but isn't exposed to SWIG.
        """
        hmm = self.hmm_object

        assert isinstance(hmm, ghmm.HMM)
        df = 0
        done_pi = False
        matrix = hmm.asMatrices()
        transition, means, initial = matrix
        for state in range(len(means)):
            # state transitions
            if check_two_nonzeroes(transition[state]):
                df += hmm.cmodel.cos * (len(means) - 1)
            # initial probs
            if initial[state] not in (0, 1):
                df += 1
                done_pi = True
            # the mean and variance of the state
            df += 2
        if done_pi:
            df -= 1
        return df

    def _score(self, obs):
        """
        Calculates the common terms of BIC and AIC.
        """
        hmm = self.hmm_object
        df = self.nparams()
        is_multivariate = isinstance(hmm.asMatrices()[1][0][0], list)

        # we need to treat observations and observation sets
        # separately
        n = 0
        if isinstance(obs, ghmm.EmissionSequence):
            n = len(obs)
            if is_multivariate:
                n = n / 2
        #         print "Found N as %f" % n
        else:
            for seq in obs:
                nn = len(seq)
                if is_multivariate:
                    nn = nn / 2
                n += nn
        #         print "Found N as %f" % n
        llg = hmm.loglikelihood(obs)
        return llg, df, n

    def BIC(self, obs=None):
        if obs is None:
            obs = self.training_data
        llg, df, n = self._score(obs)
        return -2 * llg + df * np.log(n)

    def AIC(self, obs=None):
        if obs is None:
            obs = self.training_data
        llg, df, n = self._score(obs)
        return -2 * llg + df * 2

    # def __getstate__(self):
    # 	d = self.__dict__.copy()
    # 	if self.hmm_type == "ghmm":
    # 		del d['hmm_object']
    # 	return d

    # def __setstate__(self, d):
    # 	self.__dict__.update(d)
    # 	if self.hmm_type == "ghmm":
    # 		self.to_ghmm()

    def is_multivariate(self):
        return isinstance(self.hmm_object.asMatrices()[1][0][0], list)

    def convert_data_to_sequence(self, data):
        multivariate = len(list(data[0][0])) == 2
        obs = data
        if multivariate:
            obs = [flatten_to_emission(d) for d in data]
        obs = ghmm.SequenceSet(domain, obs)
        return obs

    def from_matrix(self, matrices, training_data):
        # print "matrices", matrices
        # print "training data", training_data
        assert matrices != []
        assert training_data is not None
        A, B, pi = matrices
        m = self._from_matrix(A, B, pi)
        self.from_ghmm(m, training_data)

    def _from_matrix(self, A, B, pi):
        self.multivariate = False
        try:
            if len(B[0][0]) > 1:
                self.multivariate = True
        except TypeError:
            pass
        # print "Multivariate?", self.multivariate
        domain = ghmm.Float()
        dist = ghmm.GaussianDistribution(domain)
        if self.multivariate:
            dist = ghmm.MultivariateGaussianDistribution(domain)
        model = ghmm.HMMFromMatrices(emissionDomain=domain, distribution=dist, A=A, B=B, pi=pi)
        return model

    def from_ghmm(self, ghmm_obj, training_data):
        assert ghmm_obj is not None
        assert training_data is not None
        matrix = ghmm_obj.asMatrices()
        self.transmat = np.asarray(matrix[0])
        self.means, self.variances = zip(*matrix[1])
        self.multivariate = isinstance(self.means[0], list)
        self.nstates = len(self.means)
        self.initProb = np.asarray(matrix[2])
        self.means = np.asarray(self.means)
        self.variances = np.asarray(self.variances)
        self.hmm_object = ghmm_obj
        self.training_data = training_data
        # print type(training_data), ghmm.EmissionSequence.__type__
        if isinstance(training_data, ghmm.EmissionSequence):
            self.training_data = ghmm.SequenceSet(ghmm.Float(), [training_data])
        # else:
        # 	print "heyloyloy"
        # 	import sys;sys.stdout.flush()
        # print type(self.training_data)
        # dd = self.convert_data_to_sequence(training_data)
        # print type(training_data), training_data[:2]
        ll = ghmm_obj.loglikelihoods(self.training_data)
        # ll = 0
        # for d in training_data:
        # 	ll += ghmm_obj.loglikelihood(d)
        self.loglikelihood = ll  #ghmm_obj.loglikelihood(training_data)
        self.aic = self.AIC(self.training_data)
        self.bic = self.BIC(self.training_data)
        if not hasattr(self.hmm_object, "obs"):
            self.hmm_object.obs = self.training_data

    def to_ghmm(self):
        #     print matrices[1][0][0], np.asarray(matrices[1][0][0]).ndim
        # assert self.hmm_object == None
        # multivariate = len(means[0]) == 2
        # print "Multivariate?", multivariate
        # domain = ghmm.Float()
        # dist = ghmm.GaussianDistribution(domain)
        # if multivariate:
        #     dist = ghmm.MultivariateGaussianDistribution(domain)
        A = list(self.transmat)
        pi = list(self.initProb)
        B = list(zip(self.means, [flatten_to_emission(v) for v in self.variances]))
        self.hmm_object = self._from_matrix(A, B, pi)
        self.hmm_object.obs = self.training_data
        return self.hmm_object

    def from_R(self, RHmm_obj, training_data):
        assert RHmm_obj is not None
        assert training_data is not None
        univariate = False
        self.means = []
        self.training_data = training_data
        for m in list(RHmm_obj[0][2][3]):
            # print m, list(m)
            # to fix issue #32
            if isinstance(m, float):
                univariate = True
                self.means.append(m)
            else:
                self.means.append((m[0], m[1]))
        # self.means = [(m[0],m[1]) for m in list(RHmm_obj[0][2][3])]
        self.nstates = len(self.means)
        if univariate:
            self.variances = [v for v in RHmm_obj[0][2][4]]
        else:
            self.variances = [(tuple(v[:2]), tuple(v[2:])) for v in RHmm_obj[0][2][4]]
        assert len(self.variances) == self.nstates
        self.initProb = list(RHmm_obj[0][0])
        assert len(self.initProb) == self.nstates
        self.transmat = list(RHmm_obj[0][1])
        transmat_n = [[0] * self.nstates for i in range(self.nstates)]
        assert len(transmat_n) == self.nstates
        self.bic = RHmm_obj[2][0]
        self.aic = RHmm_obj[3][0]
        for i, n in enumerate(self.transmat):
            state_1 = i % self.nstates
            state_2 = i / self.nstates
            transmat_n[i % self.nstates][i / self.nstates] = n
        self.transmat = transmat_n
        self.loglikelihood = RHmm_obj[1][0]
        # self.Rhmm = RHmm_obj
        assert len(self.transmat) == self.nstates
        self.means = np.asarray(self.means)
        self.variances = np.asarray(self.variances)
        self.initProb = np.asarray(self.initProb)
        self.transmat = pd.DataFrame(self.transmat, index=range(self.nstates), columns=range(self.nstates))

    def _smooth_matrix(self, matrix):
        transmat = np.array(matrix)
        minimum = np.min(matrix)

        increment_h = minimum / float(len(transmat[0]))
        increment_v = minimum / float(len(transmat))
        for r, dummy in enumerate(transmat):
            for cell, value in enumerate(transmat[r]):
                # print transmat[r][cell]
                # import sys;sys.stdout.flush()
                if transmat[r][cell] == 0:
                    transmat[r][cell] = minimum
                    for i in range(len(transmat[r])):
                        if i != cell:
                            transmat[r][i] -= increment_h
                    for i in range(len(transmat)):
                        if i != r:
                            transmat[i][cell] -= increment_v
        return transmat

    # def to_hmmlearn(self):
    # 	from sklearn import hmm
    # 	from copy import deepcopy as copy

    # 	n_features = 1
    # 	try:
    # 		n_features = len(self.means[0])
    # 	except:
    # 		pass
    # 	h = hmm.GaussianHMM(n_components=self.nstates,
    # 	                    covariance_type="full",
    # 	                    startprob=self.initProb,
    # 	                    transmat=self._smooth_matrix(self.transmat))
    # 	h.n_features = n_features
    # 	h.means_ = self.means
    # 	h.covars_ = self.variances
    # 	return h

    def __str__(self):
        string = "Means:\n %s\n\n" % self.means
        string += "Variances:\n %s\n\n" % self.variances
        string += "Initial probabilities:\n %s\n\n" % self.initProb
        string += "Transition matrix:\n %s\n\n" % self.transmat
        string += "BIC:\n %s\n\n" % self.bic
        return string

    # <lift https://github.com/guyz/HMM/blob/master/hmm>

    def _mapB(self, observations):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        This method highly optimizes the running time, since all PDF calculations
        are done here once in each training iteration.
        - self.Bmix_map - computesand maps Bjm(Ot) to Bjm(t).
        '''
        self.B_map = np.zeros((self.nstates, len(observations)),
                              dtype=np.float_)
        self.Bmix_map = np.zeros((self.nstates, self.mixtures, len(observations)),
                                 dtype=np.float_)
        for j in range(self.nstates):
            for t in range(len(observations)):
                self.B_map[j][t] = self._calcbjt(j, observations[t])

    def _calcbjt(self, j, Ot):
        '''
        Helper method to compute Bj(Ot) = sum(1...M){Wjm*Bjm(Ot)}
        '''
        from scipy.stats import multivariate_normal
        # print "Mean:", self.means[j]
        # print "Covariance:", self.variances[j]
        # print "Observation:", Ot
        # print "Distance:", euclidean(self.means[j], Ot)
        np.seterr(under="warn", over="warn")
        # try:
        return multivariate_normal.pdf(Ot, self.means[j], self.variances[j])

    def entropy_rate(self):
        """
        Returns the estimated entropy rate of the Markov chain
        """
        from rpy2.robjects import r, globalenv
        from itertools import product
        import pandas as pd
        import pandas.rpy.common as com
        from scipy.special import xlogy

        r("library('DTMCPack')")
        globalenv['transmat'] = com.convert_to_r_dataframe(pd.DataFrame(self.transmat))
        stationary_dist = r("statdistr(transmat)")
        # long_as = lambda x: range(len(x))
        rate = 0
        for s1, s2 in product(range(len(self.means)), range(len(self.means))):
            p = self.transmat[s1][s2]
            rate -= stationary_dist[s1] * xlogy(p, p)
        return rate

    def viterbi(self, obs, flatten=False, flatten_input=False):
        """
        Receives a list of lists that contain emissions, returns a tuple
        containing the path and the likelihood.
        """
        data = obs
        if self.is_multivariate() and flatten_input:
            if not isinstance(obs[0], ghmm.EmissionSequence):
                print "Flattening..."
                data = [flatten_to_emission(d) for d in obs]
        data = ghmm.SequenceSet(ghmm.Float(), [d for d in obs])
        # paths, likelihoods =
        # max_likelihood = -99999999
        # best_path = None
        viterbi_path, likelihood = self.hmm_object.viterbi(data)
        if flatten:
            viterbi_path = [v[0] for v in viterbi_path]
        return viterbi_path, likelihood

    def viterbi_sequence(self, seq, flatten=True, flatten_input=False):
        ret = []
        for s in seq:
            ret.append(self.viterbi([s], flatten, flatten_input))
        return ret

    def viterbi_path(self, obs):
        """
        Calculates the most likely state path for the given observations. Returns
        a tuple of (probability, state_seq).
        """
        if self.hmm_type == "ghmm":
            return self.hmm_object.viterbi(obs)

        V = [{}]
        path = {}
        states = range(self.nstates)
        start_p = self.initProb
        trans_p = self.transmat
        self._mapB(obs)
        emit_p = self.B_map

        # Initialize base cases (t == 0)
        for y in states:
            pp = emit_p[y][0]
            V[0][y] = start_p[y] * pp
            path[y] = [y]

        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in states:
                (prob, state) = max((V[t - 1][y0] * trans_p[y0][y] * emit_p[y][t], y0) for y0 in states)
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            # Don't need to remember the old paths
            path = newpath
        n = 0  # if only one element is observed max is sought in the initialization values
        if len(obs) != 1:
            n = t
        # print_dptable(V)
        (prob, state) = max((V[n][y], y) for y in states)
        return ViterbiPath(prob=prob, state_seq=path[state])

    def format_training_dataset(self, data):
        if self.is_multivariate():
            return np.array(nest(data))
        else:
            return np.array(data)

    def get_training_dataset(self, index):
        return self.format_training_dataset(self.training_data[index])

    def iterate_training_datasets(self):
        return iter(map(self.format_training_dataset, self.training_data))

    ###########
    # PLOTTING
    ###########
    def on_pick(self, event):
        print event
        print self.plot_annotations
        if hasattr(event, '_processed'):
            return
        event._processed = True
        if event.artist in self.plot_annotations:
            self.on_pick_annotation(event)
        elif event.artist in self.plot_means:
            self.on_pick_means(event)
        from matplotlib import pyplot as plt
        plt.draw()

    def on_pick_annotation(self, event):
        print "Annotation: %s" % event.artist
        event.artist.set_visible(False)

    def on_pick_means(self, event):
        print "Mean: %s" % self.plot_means.index(event.artist)
        self.plot_annotations[self.plot_means.index(event.artist)].set_visible(True)
        print "%s" % self.plot_annotations[self.plot_means.index(event.artist)]

    def plot_hmm(self, axes=None, clr=constants.kelly_colors, transition_arrows=True,
                 prob_lists=True, legend=True, verbose=False, *args, **kwargs):
        from matplotlib import pyplot as plt
        from matplotlib.patches import FancyArrowPatch, ArrowStyle
        univariate = not self.is_multivariate()

        if univariate:
            print "Univariate HMM detected (%d states)." % len(self.means)
            means_ = [(0, m) for m in self.means]
        else:
            means_ = self.means
        covars = self.variances

        if axes is None:
            axes = plt.gca()

        self.plot_annotations = []
        self.plot_means = []
        self.plot_covars = []

        arrows = []
        colors = clr
        max_prob = 0
        for i, row in enumerate(self.transmat):
            for j, p in enumerate(row):
                # ignore self-transitions
                if i != j:
                    max_prob = max(max_prob, p)

                    #     max_prob = max(transmat.flatten())
        for i, mean in enumerate(means_):
            color = colors[i % len(colors)]
            if verbose:
                print  "MEAN:", tuple(mean)
            self.plot_means.append(axes.scatter(*tuple(mean), color=color, picker=10, label="State%i" % i))
            axes.annotate(s="%d" % i, xy=mean, xytext=(-10, -10), xycoords="data", textcoords="offset points",
                          alpha=1, bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.3))
            if verbose:
                print  "COVARS: %s" % covars[i]
            if not univariate:
                if verbose:
                    print "Drawing ellipse..."
                self.plot_covars.append(plot_cov_ellipse(covars[i], mean, alpha=.30, color=color, ax=axes))
            else:
                self.plot_covars.append(
                    axes.axhspan(mean[1] - np.sqrt(covars[i]), mean[1] + np.sqrt(covars[i]), color=color, alpha=.30))
            x0, y0 = mean
            prob_string = "$\pi = %f$" % self.initProb[i]
            for j, p in enumerate(self.transmat[i]):
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
                        alpha = min(1., (np.exp(2 * alpha / (len(self.means) * .5))) - 1)

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
                self.plot_annotations.append(
                    plt.annotate(s=prob_string, xy=mean, xytext=(0, 10), xycoords="data", textcoords="offset points",
                                 alpha=1, bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.3), picker=True,
                                 visible=True))

        if self.cid:
            plt.gcf().canvas.mpl_disconnect(self.cid)
        self.cid = plt.gcf().canvas.mpl_connect('pick_event', self.on_pick)
        if legend:
            axes.legend()
        self.plot_arrows = arrows
        if verbose:
            print "Returning from plot_hmm"
        return self.plot_annotations, self.plot_means, arrows


class ViterbiPath(object):
    state_seq = log_viterbi = prob = None

    def __init__(self, state_seq=None, prob=None, viterbi_obj=None):
        if viterbi_obj:
            try:
                self.state_seq = np.asarray(viterbi_obj[0][0], dtype=int)
                self.log_viterbi = viterbi_obj[1][0][0]
                self.prob = np.exp(viterbi_obj[2][0][0])
            except TypeError:
                self.state_seq = np.asarray(viterbi_obj[0], dtype=int)
                self.log_viterbi = viterbi_obj[1][0]
                self.prob = np.exp(viterbi_obj[2][0])
        else:
            assert state_seq is not None
            self.state_seq = state_seq
            print prob, state_seq
            import sys;

            sys.stdout.flush()
            self.prob = prob

    def __str__(self):
        s = "States: %s\n\n" % self.state_seq
        s += "Log Viterbi Score: %f\n\n" % self.log_viterbi
        s += "Log Probability: %f\n\n" % self.log_prob
        return s


def levenshtein(s1, s2):
    l1 = len(s1)
    l2 = len(s2)

    matrix = [range(l1 + 1)] * (l2 + 1)
    for zz in range(l2 + 1):
        matrix[zz] = range(zz, zz + l1 + 1)
    for zz in range(0, l2):
        for sz in range(0, l1):
            if s1[sz] == s2[zz]:
                matrix[zz + 1][sz + 1] = min(matrix[zz + 1][sz] + 1, matrix[zz][sz + 1] + 1, matrix[zz][sz])
            else:
                matrix[zz + 1][sz + 1] = min(matrix[zz + 1][sz] + 1, matrix[zz][sz + 1] + 1, matrix[zz][sz] + 1)
    # print "That's the Levenshtein-Matrix:"
    # printMatrix(matrix)
    return matrix[l2][l1]


def reconstruct_hmm(matrices, training_data):
    # print "Matrices:",matrices
    # print "Training data:", training_data[:5]
    # import sys;sys.stdout.flush()
    data = [flatten_to_emission(d) for d in training_data]
    data = ghmm.SequenceSet(ghmm.Float(), data)
    return HMM(hmm_obj=None,
               training_data=data,
               matrices=matrices,
               hmm_type='ghmm')


def reduce_hmm(hmm):
    assert isinstance(hmm, HMM)
    # print "Multivariate?", hmm.multivariate
    # print "NESTEEEED"#, nest(hmm.training_data)
    data = hmm.hmm_object.obs
    assert isinstance(data, ghmm.SequenceSet)
    if hmm.multivariate:
        data = [nest(d) for d in data]
    else:
        data = [list(d) for d in data]
    assert not isinstance(data, ghmm.EmissionSequence)
    assert not isinstance(data, ghmm.SequenceSet)
    return (reconstruct_hmm, (hmm.hmm_object.asMatrices(),
                              data))

import copy_reg

copy_reg.constructor(reconstruct_hmm)
copy_reg.pickle(HMM, reduce_hmm)