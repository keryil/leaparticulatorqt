from leaparticulator.data.functions import fromFile_p2p, fromFile


class TestP2PLogsFromTest(object):
    def setup_method(self, method):
        if method in (self.test_detectP2PData,
                      self.test_readRealData):
            self.test_file = './leaparticulator/test/test_data/P2P-160203.170804.REALDATA.1.exp.log'
        else:
            self.test_file = './leaparticulator/test/test_data/P2P-160204.121143.1.exp.log'
        print "Data file to use in the following test: {}".format(self.test_file)

    def teardown_method(self, method):
        pass

    def test_readTestData(self):
        r = fromFile_p2p(self.test_file)
        print r
        assert r

    def test_readRealData(self):
        r = fromFile_p2p(self.test_file)
        print r
        assert r

    def test_detectP2PData(self):
        assert str(fromFile_p2p(self.test_file)) == str(fromFile(self.test_file))
