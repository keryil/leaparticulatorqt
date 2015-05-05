import sys
from collections import namedtuple
from twisted.python.failure import Failure

from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
from twisted.internet import defer

from leaparticulator.p2p.server import start_server, start_client, LeapP2PServerFactory
from leaparticulator.p2p.ui.server import LeapP2PServerUI


def prep(self):
    from leaparticulator import constants

    # base.DelayedCall.debug = True
    constants.setupTest()
    self.app = QApplication.instance()
    if not self.app:
        self.app = QApplication(sys.argv)

ClientData = namedtuple(
    "ClientData",
    "theremin controller connection factory client_id client_ip".split())


class P2PTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(P2PTestCase, self).__init__(*args, **kwargs)
        self.timeout=6
        self.factories = []

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
        client_id = "test%d" % int(id)

        print "Starting client with id %s" % client_id
        theremin = start_client(
            self.app, uid=client_id)
        self.factory = factory = theremin.factory
        self.factories.append(factory)
        factory.ui.go()
        data = ClientData(theremin, theremin.controller, None, factory, client_id, client_ip)
        self.clients[id] = data
        self.reactor = theremin.reactor
        return data

    def getFactories(self):
        server = None
        clients = []
        for f in self.factories:
            if isinstance(f, LeapP2PServerFactory):
                server = f
            else:
                clients.append(f)
        return server, clients

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
        self.factories.append(self.factory)
        return self.factory

    def stopServer(self):
        factory = self.getFactories()[0]
        if not isinstance(factory.listener.result, Failure):
            factory.listener.result.stopListening()
        del factory.listener
        factory.stopFactory()

    def getClients(self):
        last_round = self.factory.session.getLastRound()
        speaker, listener = last_round.speaker, last_round.hearer
        return speaker, listener
    


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
            for factory in self.factories:
                if isinstance(factory, LeapP2PServerFactory):
                    item = "%s (%s)" % (data.client_ip, data.client_id)
                    print "Looking for: ", item
                    item = factory.ui.clientModel.findItems(item, Qt.MatchExactly)
                    print "Rows: ", factory.ui.clientModel.rowCount()
                    print "Found: ", item
                    d.callback(self.assertIsNotNone(item))
        self.reactor.callLater(.5, fn)
        return d
