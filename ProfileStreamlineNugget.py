import sys
def do_it(file_id=sys.argv[1], units=sys.argv[2], parallel=sys.argv[3], skip_phases=sys.argv[4:-1],
         prefix=sys.argv[-1]):
    from leaparticulator.notebooks.StreamlinedDataAnalysisGhmm import analyze_log_file_in_phases_by_condition
    from leaparticulator import constants
    try:
        print "skip_phase=%s" % skip_phases
        print "parallel? %s" % parallel
        print "prefix=%s" % prefix
        analyze_log_file_in_phases_by_condition(file_id, nstates=range(2,26), trials=100, iter=1000, 
                                                parallel=parallel, units=units,
                                            skip_phases=skip_phases, prefix=prefix,
                                               include_practice=False, multivariate=False)
    except Exception, err:
        print err
if __name__ == "__main__":
    do_it()