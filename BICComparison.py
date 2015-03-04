# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

import sys, os
p = os.path.expanduser("~/Dropbox/ABACUS/Workspace/LeapArticulator")
print p
sys.path.append(p)

# <codecell>

files = !ls logs/1*phase*.*
blacklist = ['123R0126514']
files = [".".join((f.split('.')[:-1])) for f in files]
files = filter(lambda x: x.split(".")[-1] in ('xy', 'amp_and_freq', 'amp_and_mel'), files)
files = filter(lambda x: not any([el in x for el in blacklist]), files)
file_to_id = lambda f: ".".join(f.split('/')[-1].split('.')[:2])
files_xy = filter(lambda x: x.split(".")[-1] in ('xy',), files)
files_amp_and_mel = filter(lambda x: x.split(".")[-1] in ('amp_and_mel',), files)
files_amp_and_freq = filter(lambda x: x.split(".")[-1] in ('amp_and_freq',), files)
# for f in files:
#     print f
# print files
ids = [file_to_id(f) for f in files]
# print ids
print len(files), "files"
print files_xy[0], files_amp_and_mel[0], files_amp_and_freq[0], files[0]
# files_xy = files_xy[:2]
# files_amp_and_freq = files_amp_and_freq[:2]
# files_amp_and_mel = files_amp_and_mel[:2]

# <codecell>

from StreamlinedDataAnalysisGhmm import unpickle_results
import pandas as pd

d = pd.read_csv("/shared/AudioData/ThereminData/surfacedata.csv", na_values=["NaN"])

# <codecell>

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

def hmm_to_pykov_chain(hmm):
    from pykov import Chain
    chain = Chain()
    alphabet = [str(i) for i in range(hmm.nstates)]
    for r, row in enumerate(hmm.transmat):
        for c, cell in enumerate(row):
            chain[str(r), str(c)] = cell
    return chain

# <codecell>

import pandas as pd
from matplotlib.pyplot import *
from rpy2.rinterface import RRuntimeError
pd.set_option('mode.chained_assignment','warn')
pd.set_option("display.max_rows", 300)
# import the surface data
d = pd.read_csv("/shared/AudioData/ThereminData/surfacedata.csv", na_values=["NaN"])


da_dict = {}
states_xy = []
bics_xy = []
hmms_xy = []
states_freq = []
bics_freq = []
hmms_freq = []
states_mel = []
bics_mel = []
hmms_mel = []
phases = []
entropies_xy = []
entropies_freq = []
entropies_mel = []
llhs_xy = []
llhs_freq = []
llhs_mel = []
total_score = []
cols = ['id', 'reversed', 'condition', 'phase', 'phase_order', 'score', 'score_n', 'total_score']
# add a 5 
for u in ('xy', 'amp_and_freq', 'amp_and_mel'):
    for c in ("entropy_%s", "nstates_%s", "bic_%s", "entropy_%s_n", "nstates_%s_n", "bic_%s_n"):
        cols.append(c % u)
    
# print cols
all_data = pd.DataFrame(index=range(len(files_xy)),columns = tuple(cols))
score_cols = ["score1st", "score2nd", "score3rd"]
series = lambda x: pd.Series(x, index=all_data.index)
normalize = lambda x: (x - np.average(x)) / np.std(x)
norm_series = lambda x: series(normalize(x))

all_data["id"] = series([file_to_id(f).split(".")[0] for f in files_xy])
all_data["phase"] = series([int(f.split(".")[-2][-1]) for f in files_xy])
all_data["phase"] = series([int(f.split(".")[-2][-1]) for f in files_xy])
all_data["phase_order"] = series([int(f.split(".")[-2][-1]) for f in files_xy])

# for f in files:
#     print f
#     print any([True if hmm is not None else False for hmm in unpickle_results(f).hmms])

hmms_xy = [pick_lowest_bic(unpickle_results(f).hmms)[0] for f in files_xy]
hmms_freq = [pick_lowest_bic(unpickle_results(f).hmms)[0] for f in files_amp_and_freq]
hmms_mel = [pick_lowest_bic(unpickle_results(f).hmms)[0] for f in files_amp_and_mel]

def process_hmm(hmm, entropies, states, bics, llhs, i=None):
    try:
        entropies.append(hmm.entropy_rate())
    except FloatingPointError, e:
        print "This row gave me an underflow during entropy shit: \n%s" % i
        print e
        entropies.append(0.99)
    except RRuntimeError, e:
        print "This row gave me a RRuntimeError during entropy shit: \n%s" % i
        print e
        entropies.append(0.99)
    except AttributeError, e:
        print "This row gave me a null pointer, probably because we have no valid HMMs:",
        print e
        entropies.append(.99)
        states.append(1)
        bics.append(40000)
        llhs.append(-20000)
        return
#     print hmm.nstates
    states.append(hmm.nstates)
    bics.append(hmm.bic)
    llhs.append(np.product(hmm.loglikelihood))

# print hmms
for i, (hmm_xy, hmm_freq, hmm_mel) in enumerate(zip(hmms_xy, hmms_freq, hmms_mel)):
    process_hmm(hmm_xy, entropies_xy, states_xy, bics_xy, llhs_xy, i)
    process_hmm(hmm_freq, entropies_freq, states_freq, bics_freq, llhs_freq, i)
    process_hmm(hmm_mel, entropies_mel, states_mel, bics_mel, llhs_mel, i)
    
# print "States:",states_xy, states_freq, states_mel     
all_data["entropy_xy"] = series(entropies_xy)
all_data["entropy_amp_and_freq"] = series(entropies_freq)
all_data["entropy_amp_and_mel"] = series(entropies_mel)
all_data["bic_xy"] = series(bics_xy)
all_data["bic_amp_and_freq"] = series(bics_freq)
all_data["bic_amp_and_mel"] = series(bics_mel)
all_data["nstates_xy"] = series(states_xy)
all_data["nstates_amp_and_freq"] = series(states_freq)
all_data["nstates_amp_and_mel"] = series(states_mel)
all_data["llh_xy"] = series(llhs_xy)
all_data["llh_amp_and_freq"] = series(llhs_freq)
all_data["llh_amp_and_mel"] = series(llhs_mel)

cond = []
rev = []
scores = []
for id, phase in zip(all_data["id"], all_data["phase"]):
#     print "ID", id
    
    # this piece of trickery makes sure we don't screw up
    # due to differences in case
    row = d[d["PP NR"] == id].index.tolist()
    if len(row) < 1:
        row = d[d["PP NR"] == id.upper()].index.tolist()
    if len(row) < 1:
        row = d[d["PP NR"] == id.lower()].index.tolist()
    if len(row) < 1:
        print id, phase
        print "FUUUUCK"
    
    row = row[0]

    cond.append(d.at[row,"Condition"])
    rev.append(d.at[row, "Reversed"])
    scores.append(d.at[row, score_cols[phase]])
    total_score.append(d.at[row, "Overall"])
    
all_data["condition"] = series(cond)
all_data["reversed"] = series(rev)
all_data["score"] = series(scores)
all_data["total_score"] = series(total_score)

# all_data["entropy_xy_n"] = norm_series(entropies_xy)
# all_data["entropy_amp_and_freq_n"] = norm_series(entropies_freq)
# all_data["entropy_amp_and_mel_n"] = norm_series(entropies_mel)
all_data["bic_xy_n"] = norm_series(bics_xy)
all_data["bic_amp_and_freq_n"] = norm_series(bics_freq)
all_data["bic_amp_and_mel_n"] = norm_series(bics_mel)
all_data["nstates_xy_n"] = norm_series(states_xy)
all_data["nstates_amp_and_freq_n"] = norm_series(states_freq)
all_data["nstates_amp_and_mel_n"] = norm_series(states_mel)
all_data["llh_xy_n"] = norm_series(llhs_xy)
all_data["llh_amp_and_freq_n"] = norm_series(llhs_freq)
all_data["llh_amp_and_mel_n"] = norm_series(llhs_mel)
all_data["score_n"] = norm_series(scores)

# fix the differences in phase among conditions
for index, row in all_data.iterrows():
    if row["condition"] == 2:
        p = row["phase"]
        if p == 1:
            all_data.at[index, "phase"] = 2
        elif p == 2:
            all_data.at[index, "phase"] = 1
        else:
            continue
#         print "Switchover! %s to %s" % (p, all_data.at[index, "phase"]) 
#         print all_data.at[index, "phase"], p
        assert all_data.at[index, "phase"] != p
    
        
all_data.to_csv("all_scores_bics_nstates_by_phase.csv")
# print all_data

colors = "Blue BlueViolet Chocolate Crimson Yellow Green DarkSlateBlue DeepPink GreenYellow DarkKhaki Olive LightGray Black".split()

    
# all_data
# figure()

# hist(states)
# figure()
# print bics_freq
# print zip(phases, states), len(states)
# print bics, len(bics)
# print da_dict

# <codecell>

# print all_data["phase"]
import pandas as pd
from matplotlib.pyplot import *
%matplotlib inline
all_data = pd.read_csv("all_scores_bics_nstates_by_phase.csv")
from matplotlib.patches import Patch
zero = all_data[all_data.phase == 0]
one = all_data[all_data.phase == 1]
two = all_data[all_data.phase == 2]
get_values = lambda x: x["nstates_amp_and_freq"].values
figure(figsize=(15,10))
mappings = ['1:1', "1:2", "2:2"]
colors = [gca()._get_lines.color_cycle.next() for i in mappings] 
# print dir(colors)
# print colors
handles = [Patch(color=c, label=m) for c,m in zip(colors, mappings)]
hist([get_values(zero),get_values(one),get_values(two)], stacked=True, color=colors)#, histtype="stepfilled")
legend(handles=handles)
# # figure()
# # hist(get_values(one), label="1:2")
# # figure()
# # hist(get_values(two), label="2:2")
# legend()
# print one
# hist(one["nstates_amp_and_freq"])
# figure()
# hist(two["nstates_amp_and_freq"])
# # %matplotlib qt
# # x = "llh"
# # y = "bic"
# # from matplotlib.pyplot import scatter
# # scat = lambda x, y: plt.scatter(all_data[x], all_data[y])
# # scat(x,y)
# # xlabel(x)
# # ylabel(y)
# import StreamlinedDataAnalysis
# print plt
# def plot_hmm(hmm):
#     StreamlinedDataAnalysis.plot_hmm(means_=hmm.means, transmat=hmm.transmat, covars=hmm.variances, initProbs=hmm.initProb)
# plot_hmm(hmms[10])

# <codecell>

from rpy2.robjects import r
from rpy2 import robjects
from rpy2.robjects import globalenv
print (all_data[all_data.phase == 0]["nstates_amp_and_freq"])

# <codecell>

# %matplotlib inline
# print all_data['llh']
# scatter(all_data["score"], all_data["llh"])
def plot_trend(target_col, source_col):
    for unit in ("xy", "amp_and_freq", "amp_and_mel"):
        figure()
        scatter(all_data[target_col], all_data["%s_%s" % (source_col, unit)])
        xlabel(target_col)
        ylabel("%s_%s" % (source_col, unit))

plot_trend("score", "llh")

figure()
plot(np.divide(range(1,100),100.), np.log(np.divide(range(1,100),100.)))
xlabel("X")
ylabel("log(X)")
print all_data["llh_xy"]
# print ids

# <codecell>


