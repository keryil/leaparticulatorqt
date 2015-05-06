__author__ = 'Kerem'

import benchmark
from leaparticulator.data.functions import refactor_old_references, \
    refactor_old_references2

# class BenchmarkFromFile(benchmark.Benchmark):
#
#     each = 3
#
#     def setUp(self):
#         self.filename = "OS1.1.exp.log"
#
#     def test_regularFromFile(self):
#         fromFile(self.filename)
#
#     def test_comprehensionFromFile(self):
#         fromFile_comp(self.filename)
#
class BenchMarkRefactorOldReferences(benchmark.Benchmark):

    each = 50

    def setUp(self):
        self.string = open("OS1.1.exp.log").readlines()

    def test_regularRefactor(self):
        for line in self.string:
            refactor_old_references(line)

    def test_regexRefactor(self):
        for line in self.string:
            refactor_old_references2(line)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")
    # could have written benchmark.main(each=50) if the
    # first class shouldn't have been run 100 times.