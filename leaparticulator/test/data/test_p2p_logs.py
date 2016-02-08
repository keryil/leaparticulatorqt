from leaparticulator.data.functions import fromFile_p2p, fromFile


class TestP2PLogsFromTest(object):
    def setup_method(self, method):
        self.test_file = './leaparticulator/test/test_data/P2P-160204.121143.1.exp.log'

    def teardown_method(self, method):
        pass

    def test_read(self):
        print fromFile_p2p(self.test_file)


class TestP2PLogsFromReal(object):
    def setup_method(self, method):
        self.test_file = './leaparticulator/test/test_data/P2P-160203.170804.REALDATA.1.exp.log'

    def teardown_method(self, method):
        pass

    def test_read(self):
        print fromFile_p2p(self.test_file)


class TestP2PLogsDetectP2P(object):
    def setup_method(self, method):
        self.test_file = './leaparticulator/test/test_data/P2P-160203.170804.REALDATA.1.exp.log'

    def teardown_method(self, method):
        pass

    def test_read(self):
        assert str(fromFile_p2p(self.test_file)) == str(fromFile(self.test_file))
