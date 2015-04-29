# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%%writefile AnalyzeOnCluster.py
from StreamlinedDataAnalysis import *


def start_cluster(working_dir):
    p = subprocess.Popen("ipcluster start".split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         cwd=working_dir)
    time.sleep(5)
    p.stdout.readline()
    return not ("Cluster is already running" in p.stdout.readline())

def stop_cluster(working_dir):
    magic = "a"
    while "probably not running" not in magic:
        p = subprocess.Popen("ipcluster stop".split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             cwd=working_dir)
        time.sleep(4)
        p.stdout.readline()
        magic = p.stdout.readline()
        print magic

def analyze_by_cluster(files, working_dir, fargs, restart_cluster=True):
    nstates, trials, iter, units = fargs
#     log_name = ("logs/automated_%s.log" % datetime.now()).replace(" ", "_")
    # %logstart log_name
    # %logon
    # files = files[16:]
    # files = ['1320149514.2']
    for i, f in enumerate(files):
    #     with open(("automated_%s.log" % datetime.now()).replace(" ", "_"), "w") as log:
    #         write = lambda x: log.write(x + "\n")
            if restart_cluster:
                while not start_cluster(working_dir):
                    print_n_flush("Cluster already running; trying to restart...")
        #             write("Cluster already running; trying to restart...")
                    stop_cluster(working_dir)
                    print_n_flush("Stopped the old cluster, hopefully...")
        #             write("Stopped the old cluster, hopefully...")

                print_n_flush("Successfully started the new cluster.")
            else:
                start_cluster()
    #         write("Successfully started the new cluster.")
#             success = False
#             while not success:
            print f
        #     if f != "132r01921514.2r":
        #         continue
        #     else:
            print_n_flush("----->Starting analysis of %s (%d out of %d)" % (f, i+1, len(files)))
#             write("----->Starting analysis of %s (%d out of %d)" % (f, i+1, len(files)))    
            start = timeit.default_timer()
#                 try:
            analyze_log_file_in_phases_by_condition(f, nstates=nstates, trials=trials, iter=iter, 
                                                    units=units, parallel=True)

            stop = timeit.default_timer()
            print_n_flush("Analysis of %s done (%f seconds)" % (f, stop-start))
#                 write("Analysis of %s done (%f seconds)" % (f, stop-start))
                
    #         break
if __name__ == "__main__":
    import sys
#     print "Cluster args received: %s" % sys.argv
    f, working_dir, trials, iter, units = sys.argv[1:6]
    nstates = map(int, sys.argv[6:])
    identifying_str = \
        """
        ***********
        Cluster arguments
        
        Log file: %s
        Working directory: %s
        Number of states: %d..%d
        Number of trials: %d
        Number of iterations: %d
        Measurement to fit to model: %s
        ***********
        """ % (f, working_dir, int(nstates[0]), int(nstates[-1]), int(trials), int(iter), units)
    print_n_flush(identifying_str)
    analyze_by_cluster([f], working_dir=working_dir, fargs=(nstates, int(trials), int(iter), units))
    stop_cluster(working_dir)

# <codecell>

# %%script ipython
from leaparticulator import constants
import subprocess, sys, time
from glob import glob
import timeit
working_dir = !pwd
working_dir = str(working_dir[0])
print working_dir
# from AnalyzeOnCluster import analyze_by_cluster


files = glob("logs/*.*.exp.log")
# print files[0]
blacklist = ["123R0126514.1r"]
path_to_cond = lambda x: x[:-8].split(".")[-1] 
file_to_id = lambda x: x.split('/')[-1][:-8]

# only files with condition codings
files = filter(lambda x: path_to_cond(x) in ('master','1','1r','2','2r'), files)
files = map(file_to_id, files)

# only files not blacklisted
files = filter(lambda x: x not in blacklist, files)

# only files with valid participant ids
files = filter(lambda x: x[0] == '1', files)
 
nstates = range(2,31)
trials = 1
iterations = 1000
units = constants.AMP_AND_MEL
# files.index("13202126514.2")
# files = ["123R0126514.1r"]
print files
for i, f in enumerate(files[1:]):
    print "\n***********\nSending %s to the cluster, file %d of %d.\n***********\n" % (f, i+1, len(files))
    nstates = " ".join(map(str, nstates))
    command = "python AnalyzeOnCluster.py %s %s %s %s %s %s" % (f, working_dir, trials, iterations, units, nstates)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print "Started new process with PID %d" % process.pid
    sys.stdout.flush()
    done = False
    while not done:
        out = process.stdout.readline()
        err = process.stderr.readline()
        printed = False
        if ("Analysis of %s done" % f) in out:
#             print "A-HA!"
            done = True
        if out != '':
            print out.rstrip()
            sys.stdout.flush()
            printed = True
        if err != '':
            print err.rstrip()
            sys.stdout.flush()
            printed = True
        if not printed:
            time.sleep(20)
    process.terminate()
    process.wait()
    print "Process %s returned with code %s" % (process.pid, process.returncode)
# analyze_by_cluster(files[2:], working_dir=working_dir, fargs=(nstates, trials, iter, Constants.AMP_AND_MEL))

# <codecell>


