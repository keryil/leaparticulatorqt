# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from leaparticulator import constants

# <codecell>

# %lprun -m GHmmWrapper analyze_log_file_in_phases_by_condition("1320116514.2", nstates=range(2,30), trials=5, iter=100, parallel=False, units=Constants.XY)

# <codecell>

from glob import glob

files = glob("logs/1*.*.exp.log")
# files = !ls logs/1*.*.exp.log
files
!pwd

# <codecell>

import subprocess
from itertools import product

directory = !pwd
files_n_units = product(files[:1], [constants.XY, constants.AMP_AND_FREQ, constants.AMP_AND_MEL])
p = subprocess.Popen("python ProfileStreamline.py".split(),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=directory)
out, err = p.communicate()
print out
print err
# for i, (f, unit) in enumerate(files_n_units):
# ff = f.split("/")[1][:-8]
#         print "Doing file/unit combination no.#%d: %s (unit: %s)" % (i, ff, unit)
#         a = analyze_log_file_in_phases_by_condition(ff, nstates=range(2,30), trials=100, iter=1000, parallel=True, units=unit)
#         del a
#         gc.collect()

