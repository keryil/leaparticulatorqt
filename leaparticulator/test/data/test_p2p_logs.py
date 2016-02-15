from leaparticulator.data.functions import fromFile_p2p, fromFile, toPandas_p2p

class TestP2PLogsFromTest(object):
    def setup_method(self, method):
        if method in (self.test_detectP2PData,
                      self.test_readRealData):
            self.test_file = './leaparticulator/test/test_data/P2P-160203.170804.REALDATA.1.exp.log'
        else:
            self.test_file = './leaparticulator/test/test_data/P2P-160204.121143.1.exp.log'
        print "Data file to use in the following test: {}".format(self.test_file)

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

class TestP2PLogsToPandas(object):
    def setup_method(self, method):
        self.test_file = './leaparticulator/test/test_data/P2P-160203.170804.REALDATA.1.exp.log'
        print "Data file to use in the following test: {}".format(self.test_file)

    def test_correctPhaseNumberTooGreat(self):
        df1, df2 = toPandas_p2p(self.test_file, nphases=10)
        for df in (df1, df2):
            assert min(df["phase"]) == 0
            assert max(df["phase"]) == 7

    def test_correctPhaseNumberTooLow(self):
        df1, df2 = toPandas_p2p(self.test_file, nphases=5)
        for df in (df1, df2):
            assert min(df["phase"]) == 0
            assert max(df["phase"]) == 7
