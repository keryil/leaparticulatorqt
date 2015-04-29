# IPython log file

from glob import glob
import traceback

from StreamlinedDataAnalysis import *
from leaparticulator import constants

working_dir = get_ipython().getoutput(u'pwd')
working_dir = str(working_dir[0])
print working_dir
files = glob("logs/*.*.exp.log")
files = [file for file in files if file[:-8].split(".")[-1] in ('master','1','1r','2','2r')]
file_to_id = lambda x: x.split('/')[-1][:-8]
files = map(file_to_id, files)

nstates = range(2,31)
trials = 100
iter = 1000
files.index("13202126514.2")
import subprocess, time
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
        time.sleep(3)
        p.stdout.readline()
        magic = p.stdout.readline()
        print magic
import timeit
from IPython.parallel import CompositeError
from rpy2.rinterface import RRuntimeError
from datetime import datetime
get_ipython().magic(u'logon')
get_ipython().magic(u'logstart')
# files = files[16:]
# files = ['1320149514.2']
for i, f in enumerate(files):
    with open(("automated_%s.log" % datetime.now()).replace(" ", "_"), "w") as log:
        write = lambda x: log.write(x + "\n")
        while not start_cluster(working_dir):
            print "Cluster already running; trying to restart..."
            write("Cluster already running; trying to restart...")
            stop_cluster(working_dir)
            print "Stopped the old cluster, hopefully..."
            write("Stopped the old cluster, hopefully...")

        print "Successfully started the new cluster."
        write("Successfully started the new cluster.")
        success = False
        while not success:
            print f
        #     if f != "132r01921514.2r":
        #         continue
        #     else:
            print "----->Starting analysis of %s (%d out of %d)" % (f, i+1, len(files))  
            write("----->Starting analysis of %s (%d out of %d)" % (f, i+1, len(files)))    
            start = timeit.default_timer()
            try:
                analyze_log_file_in_phases_by_condition(f, nstates=nstates, trials=trials, iter=iter, 
                                                        units=constants.AMP_AND_MEL, parallel=False)

                stop = timeit.default_timer()
                print "Analysis of %s done (%f seconds)" % (f, stop-start)
                write("Analysis of %s done (%f seconds)" % (f, stop-start))
            except RRuntimeError, e:
                print "RRuntimeError: %s" % e
                write("RRuntimeError: %s" % e)
            except CompositeError, e:
                print "Composite error ******"
                print e
                write("Composite error ******")
                write(e)
    #             e.raise_exception()
            except Exception, e:
                print "Error during analysis: "
                print e, e.args
                print traceback.format_exc()
                write("Error during analysis: ")
                write( e, e.args)
                write( traceback.format_exc())
            except:
                print "Some other exception that is not an exception."
                write("Some other exception that is not an exception.")
            finally:
                success = True
#         break
