__author__ = 'Kerem'
from leaparticulator.data.functions import recursive_decode, fromFile
from leaparticulator.data.frame import LeapFrame
from leaparticulator import constants
import unittest, os

class TestRecursiveUnpickle(unittest.TestCase):

    def test_fromFile(self):
        from glob import glob
        root = os.path.split(__file__)[0]
        files = glob(os.path.join(root, '..', 'test_data', '*.exp.log'))
        print files
        for f in files:
            print "Testing on %s" % os.path.split(f)[-1]
            responses, test_results, responses_practice, test_results_practice, images = fromFile(f)
            signals = responses['127.0.0.1']['0']
            frame = signals[signals.keys()[0]][0]
            self.assertIsInstance(frame, LeapFrame)

if __name__ == '__main__':
    unittest.main()