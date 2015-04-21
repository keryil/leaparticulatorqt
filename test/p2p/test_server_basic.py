import sys
from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from LeapP2PServer import start_server, start_client
from twisted.internet import defer


def prep(self):
    import Constants
    Constants.leap_server = "127.0.0.1"
    self.app = QApplication.instance()
    if not self.app:
        self.app = QApplication(sys.argv)


class P2PServerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.factory.listener.result.stopListening()
            self.factory.stopFactory()
        # self.app.quit()
        # from time import sleep
        # sleep(1)

    def setUp(self):
        prep(self)

    def test_startUp(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=False)
        self.assertIsNotNone(self.factory)

    def test_invalidCondition(self):
        self.assertRaises(
            Exception, lambda: start_server(self.app, condition=1, no_ui=False))

    def test_headlessStartup(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=True)
        self.assertIsNotNone(self.factory)
        self.assertIsNone(self.factory.ui)


class P2PServerTestWithClient(unittest.TestCase):

    def tearDown(self):
        self.factory.listener.result.stopListening()
        self.factory.stopFactory()

    def setUp(self):
        prep(self)
        self.factory = start_server(
            self.app, condition='1', no_ui=True)

    def test_connect(self):
        client_ip = "127.0.0.1"
        client_id = "test1"
        theremin, reactor, controller, connection, factory = start_client(
            self.app, uid=client_id, run=False)
        return factory.connection_def
        # defer.gatherResults([factory.connection_def])
        # item = "%s (%s)" % (client_ip, client_id)
        # item = self.factory.ui.clientModel.findItems(item, Qt.MatchExactly)
        # print item
