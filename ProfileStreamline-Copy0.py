# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

files = !ls
logs / 1 *.*.exp.log
files

# <codecell>

% % writefile
ProfileStreamlineNugget.py
import sys


def do_it(file_id=sys.argv[1], units=sys.argv[2]):
    from StreamlinedDataAnalysisGhmm import analyze_log_file_in_phases_by_condition

    try:
        analyze_log_file_in_phases_by_condition(file_id, nstates=range(2, 30), trials=100, iter=1000, parallel=False,
                                                units=units,
                                                skip_phases=[0, 1])
    except Exception, err:
        print err


if __name__ == "__main__":
    do_it()

# <codecell>

import gc
from leaparticulator import constants
from itertools import product
from subprocess import os
from ProfileStreamlineNugget import do_it

files_n_units = product(files, [constants.XY, constants.AMP_AND_FREQ, constants.AMP_AND_MEL])
dd = os.getcwd()

for i, (f, unit) in enumerate(files_n_units):
    if i < 4:
        # 3 was a problem
        continue
    ff = f.split("/")[1][:-8]
    print "Doing file/unit combination no.#%d: %s (unit: %s)" % (i, ff, unit)
    do_it(ff, unit)
    # newcode = code % (ff, unit)
    #         p = Popen(('python ProfileStreamlineNugget.py %s %s' % (ff, unit)).split(), stdout=PIPE, stderr=PIPE, cwd=dd)
    #         line = " "
    #         while line:
    # #             line = p.stdout.readline()
    # #             if not line:
    #             print line.rstrip()
    #             line = p.stdout.readline()
    #         print "ERR\n", p.stderr.readlines()#communicate()[1]
    #         print "OUT\n", std
    #         print "ERR\n", err
    gc.collect()

# <codecell>


