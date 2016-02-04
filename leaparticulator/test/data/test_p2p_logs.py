from leaparticulator.data.functions import fromFile_p2p


class TestP2PLogs(object):
    def setup_method(self, method):
        self.test_file = './leaparticulator/test/test_data/P2P-160204.121143.1.exp.log'

    def teardown_method(self, method):
        pass

    def test_read(self):
        print fromFile_p2p(self.test_file)
