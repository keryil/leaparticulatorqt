from line_profiler import LineProfiler

from leaparticulator.data.functions import *


def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()

        return profiled_func

    return inner


fname = './leaparticulator/test/test_data/P2P-160209.144607.realdata.1.exp.log'

@do_profile(follow=[process_p2p_log])
def profile_fromFile():
    fromFile(fname)


@do_profile(follow=[toPandas_p2p, process_p2p_log])
def profile_toPandas_p2p():
    toPandas_p2p(fname, nphases=10)

#     self.test_file = './leaparticulator/test/test_data/P2P-160203.170804.REALDATA.1.exp.log'
# else:
#     self.test_file = './leaparticulator/test/test_data/P2P-160204.121143.1.exp.log'
# print "Data file to use in the following test: {}".format(self.test_file)


if __name__ == "__main__":
    profile_toPandas_p2p()
