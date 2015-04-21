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
    Constants.TEST = True
    self.app = QApplication.instance()
    if not self.app:
        self.app = QApplication(sys.argv)


class P2PTestCase(unittest.TestCase):

    def startClient(self, id):
        client_ip = "127.0.0.1"
        client_id = "test%d" % id
        theremin, self.reactor, controller, connection, factory = start_client(
            self.app, uid=client_id)
        return (theremin, controller, connection, factory), (client_id, client_ip)

    def startServer(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=False)
        return self.factory

    def stopServer(self):
        self.factory.listener.result.stopListening()
        self.factory.stopFactory()


class P2PServerTest(P2PTestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.stopServer()

    def setUp(self):
        prep(self)

    def test_startUp(self):
        self.startServer()
        self.assertIsNotNone(self.factory)
        self.assertIsNotNone(self.factory.ui)

    def test_invalidCondition(self):
        self.assertRaises(
            Exception, lambda: startServer(self.app, condition=1,
                                            no_ui=False))

    def test_headlessStartup(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=True)
        self.assertIsNotNone(self.factory)
        self.assertIsNone(self.factory.ui)


class P2PServerTestWithClient(P2PTestCase):

    def tearDown(self):
        self.stopServer()

    def setUp(self):
        prep(self)
        self.startServer()

    def test_connect(self):
        stuff, (client_id, client_ip) = self.startClient(1)
        self.reactor.iterate(1)
        item = "%s (%s)" % (client_ip, client_id)
        item = self.factory.ui.clientModel.findItems(item, Qt.MatchExactly)
        self.assertTrue(item)
        self.assertEqual(item[0].text(), '%s (%s)' % (client_ip, client_id))
