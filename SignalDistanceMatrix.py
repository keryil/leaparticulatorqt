# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# %load_ext sage
import numpy as np
from scipy.spatial.distance import euclidean

from leaparticulator.data.functions import fromFile, levenshtein
from leaparticulator.data.hmm import HMM, levenshtein


files = !ls
img / meanings / * _ *.png
# print files
images_phase1 = !ls
~ / Dropbox / ABACUS / Workspace / LeapArticulator / img / meanings / 5
_ *.png
images_phase2 = !ls
~ / Dropbox / ABACUS / Workspace / LeapArticulator / img / meanings / * _[135].png
images_phase3 = images_phase2
print images_phase1


def title(m):
    return m.split("/")[-1]


def titles(file_list):
    return [title(f) for f in file_list]


def meaning_to_coordinates(meaning):
    return tuple(map(float, meaning.split("/")[-1].split(".")[0].split("_")))


def distance(m1, m2):
    # m1_1, m1_2 = m1.split("/")[-1].split(".")[0].split("_")
    #     m2_1, m2_2 = m2.split("/")[-1].split(".")[0].split("_")
    #     m1_1 = float(m1_1)
    #     m1_2 = float(m1_2)
    #     m2_1 = float(m2_1)
    #     m2_2 = float(m2_2)
    return euclidean(meaning_to_coordinates(m1), meaning_to_coordinates(m2))  #sqrt((m2_1-m1_1)**2 + (m2_2-m1_2)**2)


def distance_matrix(meanings, labels, normalize=False):
    # matrix = np.asarray([[distance(m1, m2) for m1 in meanings] for m2 in meanings])
    matrix = np.asarray([[distance(m1, m2) for m1 in meanings] for m2 in meanings])
    matrix = pd.DataFrame(matrix, index=titles(labels), columns=titles(labels))
    if normalize:
        maximum = np.max(matrix)
        matrix = np.divide(matrix, maximum)
    #     print meanings
    #     print matrix
    return matrix


def viterbi_path(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max((V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
    n = 0  # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t
    print_dptable(V)
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])


def pick_lowest_bic(hmms):
    hmm, bic = None, 99999999999999
    for h in hmms:
        if h is None:
            continue
        if h.bic < bic:
            hmm = h
            bic = h.bic
    if hmm == None:
        print f, hmms
    return (hmm, bic)


def hmm_to_sage_hmm(hmm):
    transmat = hmm.transmat
    initProb = hmm.initProb
    new = hmm.GaussianHiddenMarkovModel()

# meaning_matrix = np.asarray([[distance(f, f2) for f2 in files] for f in files])
# meaning_matrix = pd.DataFrame(meaning_matrix, index=[title(f) for f in files], columns=[title(f) for f in files])e

# <codecell>

# %load_ext rpy2.ipython
files = !ls
logs / 1 * phase *.*
blacklist = ['123R0126514']
files = [".".join((f.split('.')[:-1])) for f in files]
files = filter(lambda x: x.split(".")[-1] in ('xy', 'amp_and_freq', 'amp_and_mel'), files)
files = filter(lambda x: not any([el in x for el in blacklist]), files)
file_to_id = lambda f: ".".join(f.split('/')[-1].split('.')[:2])
files_xy = filter(lambda x: x.split(".")[-1] in ('xy',), files)
files_amp_and_mel = filter(lambda x: x.split(".")[-1] in ('amp_and_mel',), files)
files_amp_and_freq = filter(lambda x: x.split(".")[-1] in ('amp_and_freq',), files)
# for f in files:
# print f
# print files
ids = [file_to_id(f) for f in files]
# print ids
print len(files), "files"
print files_xy[0], files_amp_and_mel[0], files_amp_and_freq[0]
# print ids

# <codecell>

from rpy2.robjects import r, globalenv, FloatVector


def is_multivariate(hmm):
    try:
        #         print hmm.means
        iter(hmm.means[0])
        return True
    except:
        return False


def r_list_of_stuff(arr, func, name):
    r("%s <- list()" % name)
    for i, v in enumerate(arr):
        globalenv["tmp"] = func(v)
        r("%s[[%d]] <- tmp" % (name, i + 1))
    del globalenv["tmp"]


def rhmm_dist_from_hmm(hmm):
    print "rhmm_dist_from_hmm"
    mean = np.asarray(hmm.means)
    cov = np.asarray(hmm.variances)
    #     print cov

    command = "dist <- distributionSet(dis='NORMAL', mean=%s, cov=%s)"

    globalenv['mean'] = mean
    globalenv['covar'] = cov

    if is_multivariate(hmm):
        f = lambda x: r.matrix(FloatVector(np.asarray(x).flatten()), nrow=len(x))
        r_list_of_stuff(hmm.variances, func=f, name="covar")
        f = lambda x: FloatVector(np.asarray(x))
        r_list_of_stuff(hmm.means, func=f, name="mean")
        #         r("print(covar)")
        #         r("print(mean)")
        command = "dist <- distributionSet(dis='NORMAL', mean=mean, cov=covar)"
    else:
        command = "dist <- distributionSet(dis='NORMAL', mean=as.vector(mean), var=as.vector(covar))"
    #     print command
    r(command)
    return globalenv['dist']


def rhmm_from_hmm(hmm):
    print "rhmm_from_hmm"
    from rpy2.robjects import r, globalenv

    r("rm(list=ls())")
    r("source('SampleHMM.R');library('RHmm')")
    hmm_ = hmm
    dist = rhmm_dist_from_hmm(hmm_)
    globalenv['initProb'] = np.asarray(hmm_.initProb)
    globalenv['transMat'] = np.asarray(hmm_.transmat)
    command = "hmm <- HMMSet(initProb=as.vector(initProb), transMat=as.matrix(transMat), distribution=dist)"
    #     print command
    r(command)
    #     print r("print(hmm)")
    return globalenv['hmm']


def test_rhmm_from_hmm_univariate():
    prep = """
            source("SampleHMM.R");
            list[hmm, d] <- fitHMMtoPhase(file_id="1230105514.master", nStates=9, iter=1000, phase=0, take_vars=c(%s));
            """
    r(prep % ("'x'"))
    hmm = globalenv['hmm']
    a = rhmm_from_hmm(HMM(hmm, [1, 2, 3, 4]))
    assert str(hmm[0]) == str(a)
    del globalenv['hmm']


def test_rhmm_from_hmm_multivariate():
    prep = """
            source("SampleHMM.R");
            list[hmm, d] <- fitHMMtoPhase(file_id="1230105514.master", nStates=9, iter=1000, phase=1, take_vars=c(%s));
            """
    r(prep % ("'x','y'"))
    hmm = globalenv['hmm']
    a = rhmm_from_hmm(HMM(hmm, [1, 2, 3, 4]))
    assert str(hmm[0]) == str(a)
    del globalenv['hmm']


# <codecell>

% load_ext
ipython_nose
% nose

# <codecell>

% % R
source("SampleHMM.R");
list[hmm, d] < - fitHMMtoPhase(file_id="1230105514.master", nStates=9, iter=1000, phase=1, take_vars=c("x"));
# print(hmm)
obs < - as.vector(d[1])
vv < - viterbi(hmm, obs)

# <codecell>

from StreamlinedDataAnalysis import unpickle_results
from itertools import product
from skbio.stats.distance import mantel, DistanceMatrix
# from BICComparison import pick_lowest_bic

ids = list(set(ids))
to_log = lambda x: "logs/" + x + ".exp.log"
to_hmm_file = lambda id, phase, units: to_log(id) + ".phase" + phase + "." + units
hmm = None


# for each participant
for id in ids[7:8]:
    print id
    responses, tests, responses_t, tests_t, images = fromFile(to_log(id))
    responses, tests, responses_t, tests_t, images = responses['127.0.0.1'], tests['127.0.0.1'], responses_t[
        '127.0.0.1'], tests_t['127.0.0.1'], images
    # for each phase
    for phase in responses.keys():
        print "Phase", phase
        # find the relevant images
        image_list = images_phase1
        if phase == "1":
            image_list = images_phase2
        elif phase == "2":
            image_list = images_phase3
        else:
            if phase != "0":
                print "What the fuck is a phase %s?!?" % phase

        # compute the meaning distance matrix
        meaning_matrix = distance_matrix(image_list, image_list, normalize=False)

        #         trajectory_matrix_by_prob = meaning_matrix.copy()
        trajectory_matrix_by_stateseq = meaning_matrix.copy()
        viterbis = {}
        # for each image
        for image in responses[phase].keys():

            # find and fetch the relevant hmm
            hmm_file = to_hmm_file(id, phase, "amp_and_freq")
            results = unpickle_results(hmm_file)
            hmm, bic = pick_lowest_bic(results.hmms)
            # ...and send it to R
            #             rhmm_from_hmm(hmm)

            # find the relevant trajectory
            trajectory = responses[phase][image]
            command = None
            print "Is multivariate?", is_multivariate(hmm)
            if is_multivariate(hmm):
                trajectory = [frame.get_stabilized_position()[:2] for frame in trajectory]
            #                 trajectory = [FloatVector(frame.get_stabilized_position()[:2]) for frame in trajectory]
            #                 globalenv['trajectory'] = ListVector(trajectory)
            else:
                trajectory = [frame.get_stabilized_position()[0] for frame in trajectory]
            #                 trajectory = [frame.get_stabilized_position()[0] for frame in trajectory]
            #                 globalenv['trajectory'] = FloatVector(trajectory)
            #             print trajectory
            # compute the corresponding state sequence            
            #             globalenv['trajectory'] = ListVector(trajectory)
            #             print trajectory
            #             command = "viterbi(hmm, trajectory)"

            #             viterbi = r(command)
            #             print pd.DataFrame(hmm.means)
            #             print np.asarray(hmm.means).shape
            #             print np.asarray(hmm.means).shape[1]
            result = hmm.to_hmmlearn().predict(trajectory)
            viterbis[title(image)] = result
        #             print ViterbiPath(viterbi)
        images = titles(responses[phase])
        maximum = 0
        for image1, image2 in product(images, images):
            #             print image1, image2
            dist = levenshtein(viterbis[image1], viterbis[image2])
            trajectory_matrix_by_stateseq[image1][image2] = dist
            maximum = max((maximum, dist))
        trajectory_matrix_by_stateseq = np.divide(trajectory_matrix_by_stateseq, maximum)
        #         print "Meaning matrix:"
        #         print meaning_matrix
        #         print "Signal matrix:"
        #         print trajectory_matrix_by_stateseq
        #         print meaning_matrix.T == meaning_matrix
        #         print trajectory_matrix_by_stateseq.T == trajectory_matrix_by_stateseq
        d1 = DistanceMatrix(meaning_matrix)
        d2 = DistanceMatrix(trajectory_matrix_by_stateseq)
        coef = mantel(d1, d2, method="spearman")
        #         print d1
        #         print d2
        print "Correlation coefficient:", coef[0], "(p=%f)" % coef[1]
        import sys;

        sys.stdout.flush()
        break
#         from time import sleep;sleep(1)

# <codecell>

% load_ext
sage

states = 3
transmat = np.asarray([[.2, .3, .5], [.3, .6, .1], [.5, .3, .2]])
initProb = np.asarray([.3, .3, .4])
means = np.asarray([(i, i) for i in range(states)])
spreads = np.asarray([[[i, i], [i, i]] for i in range(states)])
print "Transmat", transmat
print "Init", initProb
print "Mean", means
print "Spread", spreads
x = [[mean, spread] for mean, spread in zip(means, spreads)]
print x
hmm.GaussianHiddenMarkovModel(transmat, x, initProb)

# <codecell>

from LeapTheremin import palmToAmpAndFreq, freqToMel
import numpy as np

# <codecell>

grid_x = np.linspace(-250, 250, 30)
grid_y = np.linspace(1, 500, 30)
grid_z = grid_y

# <codecell>

from itertools import product

f = figure()
x, y, z = zip(*[i for i in product(grid_x, grid_y, grid_z)])
scatter(x, y, s=2, color='blue')
f.suptitle('XY Projection')
figure().suptitle('Amplitude/Frequency Projection')
y, x = zip(*[palmToAmpAndFreq(i) for i in product(grid_x, grid_y, grid_z)])
scatter(x, y, s=2, color='blue')

figure().suptitle('Amplitude/MelFrequency Projection')
x = [freqToMel(xx) for xx in x]
scatter(x, y, s=2, color='blue')

# <codecell>

% matplotlib
qt
from matplotlib import pylab

pylab.figure()
id = "132R0128514.2r"
responses, tests, responses_t, tests_t, images = fromFile(to_log(id))
responses = responses['127.0.0.1']
hmm_file = to_hmm_file(id, phase, "amp_and_freq")
results = unpickle_results(hmm_file)
hmm, bic = pick_lowest_bic(results.hmms)
print hmm.means
reload(HMMPlotting)
HMMPlotting.plot_hmm(hmm.means, hmm.transmat, hmm.variances)
# data = [f.get_stabilized_position() for f in responses['1']["./img/meanings/3_5.png"]]
data = HMMPlotting.to_trajectory_object([v for v in responses['1'].values()], units="amp_and_freq")
# print data
HMMPlotting.plot_trajectories(data)
pylab.show()

# <codecell>

results = unpickle_results("logs/132R0128514.2r.exp.log.phase1")

# <codecell>

responses, test_results, responses_practice, test_results_practice, images = fromFile("logs/132R0128514.2r.exp.log")

# <codecell>

import pandas as pd

len(responses['127.0.0.1']['1']['./img/meanings/1_1.png'])

# <codecell>


