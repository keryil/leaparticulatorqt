import sys
from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from LeapP2PServer import start_server, start_client
from twisted.internet import defer, base
from collections import namedtuple


def prep(self):
    import Constants

    # base.DelayedCall.debug = True
    Constants.setupTest()
    self.app = QApplication.instance()
    if not self.app:
        self.app = QApplication(sys.argv)

ClientData = namedtuple(
    "ClientData",
    "theremin controller connection factory client_id client_ip".split())


class P2PTestCase(unittest.TestCase):
    def runTest(self):
        pass

    def click(self, widget):
        print "Left clicking %s" % str(widget)
        QTest.mouseClick(widget, Qt.LeftButton)

    def startClient(self, id=None):
        if not hasattr(self, "clients"):
            self.clients = {}
        if id is None:
            from random import randint
            id = randint(0,10000)
        client_ip = "127.0.0.1"
        client_id = "test%d" % id
        theremin, self.reactor, controller, connection, factory = start_client(
            self.app, uid=client_id)
        factory.ui.go()
        data = ClientData(theremin, controller, connection, factory, client_id, client_ip)
        self.clients[id] = data
        return data 

    def stopClient(self, id):
        del self.clients[id]

    def stopClients(self):
        for id in list(self.clients):
            del self.clients[id]

    def startClients(self, qty):
        res = []
        for i in range(qty):
            res.append(self.startClient(i))
        return res

    def startServer(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=False)
        return self.factory

    def stopServer(self):
        self.factory.listener.result.stopListening()
        self.factory.stopFactory()


class ServerTest(P2PTestCase):

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


class ServerTestWithClient(P2PTestCase):

    def tearDown(self):
        self.stopClients()
        self.stopServer()

    def setUp(self):
        prep(self)
        self.startServer()

    def test_connect(self):
        data = self.startClient(1)
        d = defer.Deferred()
        def fn():
            item = "%s (%s)" % (data.client_ip, data.client_id)
            print "Looking for: ", item
            item = self.factory.ui.clientModel.findItems(item, Qt.MatchExactly)
            print "Rows: ", self.factory.ui.clientModel.rowCount()
            print "Found: ", item
            d.callback(self.assertTrue(item))
        self.reactor.callLater(.1,fn)
        return d
