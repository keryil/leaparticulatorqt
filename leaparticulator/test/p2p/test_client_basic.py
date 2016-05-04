from leaparticulator.p2p.client import start_client
from leaparticulator.p2p.server import start_server
from test_server_basic import P2PTestCase
from test_server_basic import prep


class P2PClientTest(P2PTestCase):

    # def tearDown(self):
    #     if hasattr(self, 'factory'):
    #         self.factory.hearer.result.stopListening()
    #         self.factory.stopFactory()
        # self.app.quit()

    def setUp(self):
        prep(self)
        # self.factory = start_server(
            # self.app, condition='1', no_ui=False)

    def test_startUp(self):
        theremin = start_client(
            self.app, uid="test1")
        self.assertIsNotNone(theremin)
        self.assertIsNotNone(theremin.factory)

    def test_noUID(self):
        self.assertRaises(Exception, lambda : start_client(
            self.app, uid=None))


class ClientTestWithServer(P2PTestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.server_factory.listener.result.stopListening()
            self.server_factory.stopFactory()

    def setUp(self):
        prep(self)
        self.server_factory = start_server(
            self.app, condition='1', no_ui=False)

    def test_connect(self):
        theremin = start_client(
            self.app, uid="test1")
        return theremin.factory.connection_def
