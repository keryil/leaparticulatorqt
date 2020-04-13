import os
import re

import leaparticulator.constants as Constants
from leaparticulator.constants import palmToAmpAndFreq, palmToAmpAndMel


def first_exp_parse_filename(fname):
    fname = os.path.basename(fname)
    parsed = re.search("(.+)\.(.+)\.exp\.log\.phase(\d)\.(.+).hmms", fname)
    return parsed.groups()[1:]


def first_experiment(fname):
    condition, phase, units = first_exp_parse_filename(fname)
    print("Condition: %s, phase: %s, units: %s" % (condition, phase, units))
    multivariate = False
    reverse_cond = condition in ("2r", "1r")
    interval = 1
    pick_var = 0
    if reverse_cond:
        interval = -1
        pick_var = 1

    if multivariate is None:
        if condition in ("2", "2r"):
            if phase == 1:
                multivariate = True
        else:
            if phase == 2:
                multivariate = True

    formatData = None

    if multivariate:
        if units == Constants.XY:
            formatData = lambda traj: [t[:2][::interval] for t in traj.data]
        elif units == Constants.AMP_AND_FREQ:
            # -interval, because amp_and_freq returns y,x and not x,y.
            formatData = lambda traj: [palmToAmpAndFreq(t)[::-interval] for t in traj.data]
        elif units == Constants.AMP_AND_MEL:
            # -interval, because amp_and_freq returns y,x and not x,y.
            formatData = lambda traj: [palmToAmpAndMel(t)[::-interval] for t in traj.data]
    else:
        if units == Constants.XY:
            formatData = lambda traj: [t[pick_var] for t in traj.data]
        elif units == Constants.AMP_AND_FREQ:
            # -interval, because amp_and_freq returns y,x and not x,y.
            formatData = lambda traj: [palmToAmpAndFreq(t[pick_var]) for t in traj.data]
        elif units == Constants.AMP_AND_MEL:
            # -interval, because amp_and_freq returns y,x and not x,y.
            formatData = lambda traj: [palmToAmpAndMel(t[pick_var]) for t in traj.data]

    return formatData
