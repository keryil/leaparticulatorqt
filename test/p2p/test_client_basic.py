from leaparticulator.p2p.server import start_server, start_client
from test_server_basic import prep
from test_server_basic import P2PTestCase


class P2PClientTest(P2PTestCase):

    # def tearDown(self):
    #     if hasattr(self, 'factory'):
    #         self.factory.listener.result.stopListening()
    #         self.factory.stopFactory()
        # self.app.quit()

    def setUp(self):
        prep(self)
        # self.factory = start_server(
            # self.app, condition='1', no_ui=False)

    def test_startUp(self):
        theremin, reactor, controller, connection, factory = start_client(
            self.app, uid="test1")
        self.assertIsNotNone(theremin)
        self.assertIsNotNone(controller)
        self.assertIsNotNone(factory)

    def test_noUID(self):
        self.assertRaises(Exception, lambda : start_client(
            self.app, uid=None))


class ClientTestWithServer(P2PTestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.factory.listener.result.stopListening()
            self.factory.stopFactory()

    def setUp(self):
        prep(self)
        self.factory = start_server(
            self.app, condition='1', no_ui=False)

    def test_connect(self):
        theremin, reactor, controller, connection, factory = start_client(
            self.app, uid="test1")
        return factory.connection_def
