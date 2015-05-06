
# coding: utf-8

# In[6]:

from StreamlinedDataAnalysisGhmm import analyze_log_file_in_phases_by_condition
from os import getcwd, chdir
from os.path import expanduser, sep
from leaparticulator import constants
# move to root dir
chdir(expanduser("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt/"))
getcwd()


# In[12]:

from glob import glob
prefix = 'logs/logs/orange_squares/'
files = glob(prefix + "*.exp.log")
nstates = range(2,26)
trials=
print files


# In[11]:

for f in files:
    for unit in (constants.AMP_AND_FREQ, constants.XY, constants.AMP_AND_MEL):
        file_id = '.'.join(f.split(sep)[-1].split(".")[:-2])
        print "File: %s" % file_id
        analyze_log_file_in_phases_by_condition(file_id, nstates, trials, iter, units=unit, parallel=True, 
                                            prefix=prefix, skip_phases=[])


# In[ ]:



