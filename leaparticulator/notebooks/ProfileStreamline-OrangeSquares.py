
# coding: utf-8

# In[1]:

from os import getcwd, chdir
from os.path import expanduser, join
from glob import glob
chdir(expanduser("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt"))
prefix = "logs/logs/orange_squares"
files = glob(join(prefix,'*.*.exp.log'))
# files = files[:2]
# files = map(lambda x: "logs/discrete/%s.exp.log" % x, "D13200321014.2".split(","))
files


# In[2]:

get_ipython().run_cell_magic(u'writefile', u'ProfileStreamlineNugget.py', u'import sys\ndef do_it(file_id=sys.argv[1], units=sys.argv[2], parallel=sys.argv[3], skip_phases=sys.argv[4:],\n         prefix=sys.argv[5:]):\n    from leaparticulator.notebooks.StreamlinedDataAnalysisGhmm import analyze_log_file_in_phases_by_condition\n    from leaparticulator import constants\n    try:\n        print "skip_phase=%s" % skip_phases\n        print "parallel? %s" % parallel\n        analyze_log_file_in_phases_by_condition(file_id, nstates=range(2,26), trials=100, iter=1000, \n                                                parallel=parallel, units=units,\n                                            skip_phases=skip_phases, prefix=prefix)\n    except Exception, err:\n        print err\nif __name__ == "__main__":\n    do_it()')


# In[ ]:

import warnings, sys
warnings.filterwarnings('ignore')
from leaparticulator.notebooks.StreamlinedDataAnalysisGhmm import analyze_log_file_in_phases_by_condition
from leaparticulator import constants #, GHmmWrapper, gc
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
#         if i <= 19:
#             continue
        print f
        ff = f.split("/")[-1][:-8]
        cond = ff.split(".")[-1]
        if cond == "master":
            cond = "1"
        multivariate = ("1" in cond and phase==2) or ("2" in cond and phase==1)
#         if i not in [11,14,17]:
#             continue
        if i in skip: #or not multivariate:
            continue
        print "Multivariate?", multivariate
#         if i in skip or i % 3 == 2:#< 11:# or i > 5:
#             # 3 (11) was a problem
#             continue
        status = "Doing file/unit/phase combination no.#%d/%d: %s, phase%s (unit: %s) (cond: %s)" % (i+1, len(files)*3*3, ff, phase, unit, cond)
        print status
        log_file.write(status)
#         do_it(ff, unit)
#         newcode = code % (ff, unit)
        skip_phase = map(str, list(set(range(3)) - set([phase])))
#         do_it(ff,unit,False,skip_phase)
#         p = Popen(('python ProfileStreamlineNugget.py %s %s %s %s' % (ff, 
#                                                                       unit,
#                                                                       True,
#                                                                       " ".join(skip_phase))).split(), 
#                                                                       stdout=PIPE, 
#                                                                       stderr=STDOUT,
#                                                                       cwd=dd)
        do_it(ff, unit, True, skip, prefix=prefix)
#         line = " "
#         while line:
# #             line = p.stdout.readline()
# #             if not line:
#             output.append(line)
# #             print line.rstrip()
#             if "GHMM" not in str(line):
#                 print line.rstrip()
#             log_file.write(line)
            
# #             else:
# #                 warning.append(line)
#             line = p.stdout.readline()
            
# #             line += "\n" + p.stderr.readline()
#             # this prevents multivariate models from failing
#             # when in cluster mode for whatever reason
# #             if p.stderr:
# #                 a = "%s" % p.stderr.readlines()
# #                 del a
#             sys.stdout.flush()
#         print "Return code:", p.returncode
#         error = p.stderr.readlines()#communicate()[1]
#         print "OUT\n", std
#         print "ERR\n", err
#         gc.collect()


# In[ ]:




# In[ ]:




# In[ ]:



