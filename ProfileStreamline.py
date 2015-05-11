# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

files = !ls
logs / discrete / *.*.exp.log
# files = files[:2]
files

# <codecell>

% % writefile
ProfileStreamlineNugget.py
import sys


def do_it(file_id=sys.argv[1], units=sys.argv[2], parallel=sys.argv[3], skip_phases=sys.argv[4:]):
    from StreamlinedDataAnalysisGhmm import analyze_log_file_in_phases_by_condition

    try:
        print "skip_phase=%s" % skip_phases
        print "parallel? %s" % parallel
        analyze_log_file_in_phases_by_condition(file_id, nstates=range(2, 31), trials=100, iter=1000,
                                                parallel=parallel, units=units,
                                                skip_phases=skip_phases, prefix="logs/discrete")
    except Exception, err:
        print err


if __name__ == "__main__":
    do_it()

# <codecell>

import warnings, sys

warnings.filterwarnings('ignore')
from leaparticulator import constants
from itertools import product
from subprocess import Popen, PIPE, os, STDOUT
from ProfileStreamlineNugget import do_it

files_n_units = product(files, [constants.XY, constants.AMP_AND_FREQ, constants.AMP_AND_MEL], range(3))
dd = os.getcwd()
error = []
warning = []
output = []
# skip = [11,14, 17]
# skip = lrange(115,226)
skip = []
# 11 is a problem, again

# skip = range(207)
# return
p = None
from datetime import datetime

log_file = open("logs/StreamlineLog.%s.log" % datetime.now(), 'w', 0)
for i, (f, unit, phase) in enumerate(files_n_units):
    ff = f.split("/")[-1][:-8]
    cond = ff.split(".")[-1]
    if cond == "master":
        cond = "1"
    multivariate = ("1" in cond and phase == 2) or ("2" in cond and phase == 1)
    # if i not in [11,14,17]:
    #             continue
    if i in skip:  #or not multivariate:
        continue
    print "Multivariate?", multivariate
    #         if i in skip or i % 3 == 2:#< 11:# or i > 5:
    #             # 3 (11) was a problem
    #             continue
    status = "Doing file/unit/phase combination no.#%d/%d: %s, phase%s (unit: %s) (cond: %s)" % (
    i + 1, len(files) * 3 * 3, ff, phase, unit, cond)
    print status
    log_file.write(status)
    #         do_it(ff, unit)
    #         newcode = code % (ff, unit)
    skip_phase = map(str, list(set(range(3)) - set([phase])))
    #         do_it(ff,unit,False,skip_phase)
    p = Popen(('python ProfileStreamlineNugget.py %s %s %s %s' % (ff,
                                                                  unit,
                                                                  True,
                                                                  " ".join(skip_phase))).split(),
        stdout=PIPE,
        stderr=STDOUT,
        cwd=dd)
    line = " "
    while line:
        #             line = p.stdout.readline()
        #             if not line:
        output.append(line)
        #             print line.rstrip()
        if "GHMM" not in str(line):
            print line.rstrip()
        log_file.write(line)

        #             else:
        #                 warning.append(line)
        line = p.stdout.readline()

        #             line += "\n" + p.stderr.readline()
        # this prevents multivariate models from failing
        # when in cluster mode for whatever reason
        #             if p.stderr:
        #                 a = "%s" % p.stderr.readlines()
        #                 del a
        sys.stdout.flush()
    print "Return code:", p.returncode
# error = p.stderr.readlines()#communicate()[1]
#         print "OUT\n", std
#         print "ERR\n", err
#         gc.collect()

# <codecell>

# list(files_n_units)
p.stderr.readlines()
# list(files_n_units).index(('logs/1320149514.2.exp.log', 'amp_and_mel', 1))

# <codecell>

import StreamlinedDataAnalysisGhmm

reload(StreamlinedDataAnalysisGhmm)

# <codecell>

# !pwd
# r = unpickle_results("logs/1230105514.master.exp.log", phase=2, units=Constants.AMP_AND_FREQ)
from leaparticulator.data import functions

responses, test_results, responses_practice, test_results_practice, images = functions.fromFile(
    "logs/1230105514.master.exp.log")

# <codecell>

from LeapTheremin import palmToAmpAndFreq

extract_velocity = lambda x: palmToAmpAndFreq(x.hands[0].palm_velocity)[::-1]
formatData = lambda r, phase: [[(palmToAmpAndFreq(frame.get_stabilized_position())[::-1],
                                 extract_velocity(frame)) for frame in rr] for rr in
                               r["127.0.0.1"][str(phase)].values()]
# formatData = lambda r, phase: [[(frame.get_stabilized_position()[:2], extract_velocity(frame)) for frame in rr] for rr in r["127.0.0.1"][str(phase)].values()]
data = zip(formatData(responses, 2), formatData(responses_practice, 2))
# velocity = [n.hands[0].palm_velocity for n in responses['127.0.0.1']['0']["./img/meanings/5_1.png"]]

# <codecell>

% matplotlib
inline
from matplotlib import pyplot as plt
import numpy as np
import StreamlinedDataAnalysisGhmm

for dat, dat_p in data:
    StreamlinedDataAnalysisGhmm.plot_quiver2d([d[0] for d in dat], C=["black" for i in d], alpha=.5)
    plt.figure()
    #     StreamlinedDataAnalysisGhmm.plot_quiver2d([d[1][:2] for d in dat], C=["red" for i in d], alpha=.5)
    #     StreamlinedDataAnalysisGhmm.plot_quiver2d(d2, C=["red" for i in d], alpha=.5)
    acc = []
    l = len(dat)
    plt.figure()
    for t0, t1 in zip(range(l), range(1, l)):
        acc.append(np.asarray(dat[t1][1][:2]) - np.asarray(dat[t0][1][:2]))
    #         acc.append()
    #     acc = [d1[1][:2]-d0[1][:2] for d0, d1 in zip([None,[0,0]] + dat, dat + [None,[0,0]])]
    StreamlinedDataAnalysisGhmm.plot_quiver2d(acc, C=["green" for i in acc], alpha=.5)
#     plt.figure()

