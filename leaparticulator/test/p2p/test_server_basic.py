import sys

from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import unittest

from leaparticulator.p2p.server import start_server, start_client, LeapP2PServerFactory, LeapP2PClientFactory


def prep(self):
    from leaparticulator import constants

    # base.DelayedCall.debug = True
    constants.setupTest()
    self.app = QApplication.instance()
    if not self.app:
        self.app = QApplication(sys.argv)


class ClientData(object):
    def __init__(self, theremin, controller, connection, factory,
                 client_id, client_ip, deferred):
        self.theremin = theremin
        self.controller = controller
        self.connection = connection
        self.factory = factory
        self.client_id = client_id
        self.client_ip = client_ip
        self.deferred = deferred

    def __str__(self):
        vars = dir(self)
        vars = filter(lambda x: not x.startswith("_"), vars)
        string = repr(self)
        for var in vars:
            string = "%s\n%s=%s" % (string, var, getattr(self, var))
        return string


class P2PTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(P2PTestCase, self).__init__(*args, **kwargs)
        self.timeout=6
        self.factories = []
        self.clients = {}
        from leaparticulator import constants
        constants.leap_server = "127.0.0.1"

    def tearDown(self):
        self.stopServer()
        for client in self.clients.values():
            print client
            client.factory.stopFactory()
        self.clients = {}
        self.server_factory = None

    def setUp(self):
        """
        Sets up two clients and a server, and initiates the
        experiment.
        :return:
        """
        from twisted.internet import reactor
        self.reactor = reactor
        prep(self)
        self.startServer()
        self.timeout = 5

        clients = [client.namedtuple for client in self.startClients(2)]
        d = defer.Deferred()

        def fn(*args):
            """
            Clicks the first screen's okay button for all clients.
            :param args:
            :return:
            """
            for client in clients:
                self.clients[client.client_id] = client
                print "Setting up client:", client
                button = client.factory.ui.firstWin.findChildren(
                        QtGui.QPushButton, "btnOkay")[0]
                self.click(button)
            d.callback(("Setup done"))

        self.reactor.callLater(.2, fn)
        # clients[0].deferred.addCallback(fn)
        # d.chainDeferred(clients[1].deferred)
        # d.addCallback(fn)
        return d  #

    def click(self, widget):
        print "Left clicking %s" % str(widget)
        QTest.mouseClick(widget, Qt.LeftButton)

    def startClient(self, id=None):
        if id is None:
            from random import randint
            id = randint(0,10000)
        client_ip = "127.0.0.1"
        client_id = "test%d" % int(id)

        print "Starting client with id %s" % client_id
        theremin = start_client(
            self.app, uid=client_id)
        factory = theremin.factory
        d = theremin.factory.connection_def
        # theremin.factory.connection_def.chainDeferred(d)  # endpoint.connect(factory)
        print "Connection deferred:", d
        self.factories.append(factory)
        assert isinstance(factory, LeapP2PClientFactory)

        data = ClientData(theremin=theremin,
                          controller=theremin.controller,
                          deferred=d,
                          factory=factory,
                          client_id=client_id,
                          client_ip=client_ip,
                          connection=None)

        def fn(client):
            data.connection = client
            # factory.ui.setClient(client)
            factory.ui.go()

        d.addCallback(fn)
        self.clients[id] = data
        data.deferred.namedtuple = data
        # from twisted.internet import reactor
        # self.reactor = reactor
        return d

    def getFactories(self):
        server = None
        clients = []
        for f in self.factories:
            if isinstance(f, LeapP2PServerFactory):
                server = f
            else:
                clients.append(f)
        assert server == self.server_factory
        return server, clients

    def stopClient(self, id):
        del self.clients[id]

    def stopClients(self):
        for id in list(self.clients):
            del self.clients[id]

    def getLastRound(self):
        return self.server_factory.session.getLastRound()

    def getRound(self, rnd_no):
        return self.server_factory.session.round_data[rnd_no]

    def startClients(self, qty):
        res = []
        for i in range(qty):
            res.append(self.startClient(i))
        return res

    def startServer(self):
        self.server_factory = start_server(
            self.app, condition='1', no_ui=False)
        self.factories.append(self.server_factory)
        return self.server_factory

    def stopServer(self):
        factory = self.server_factory
        if not isinstance(factory.listener.result, Failure):
            factory.listener.result.stopListening()
        del factory.listener
        factory.stopFactory()

    def getClientsAsServerConnections(self, rnd_no=None):
        round = None
        if rnd_no is None:
            round = self.getLastRound()
        else:
            round = self.getRound(rnd_no=rnd_no)
        speaker, listener = round.speaker, round.hearer
        return speaker, listener

    def getClientsAsClientData(self, rnd_no=None):
        speaker, listener = self.getClientsAsServerConnections(rnd_no)
        speaker_id, listener_id = [c.factory.clients[c] for c in (speaker, listener)]
        return self.clients[speaker_id], self.clients[listener_id]

    def getClientsAsUi(self, rnd_no=None):
        speaker, listener = self.getClientsAsClientData(rnd_no)
        return speaker.factory.ui, listener.factory.ui

    def create_signal(self, callback):
        ui_speaker, ui_listener = self.getClientsAsUi(rnd_no=0)

        submit_btn = ui_speaker.creationWin.findChildren(
                QtGui.QPushButton, "btnSubmit")[0]
        record_btn = ui_speaker.creationWin.findChildren(
                QtGui.QPushButton, "btnRecord")[0]

        image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]
        # record something
        from leaparticulator.data.frame import generateRandomSignal
        self.click(record_btn)
        ui_speaker.theremin.last_signal = generateRandomSignal(2)
        self.click(record_btn)

        self.click(submit_btn)
        self.reactor.callLater(.5, callback)

    def answer_question(self, answer=0, callback=None):
        # speaker, listener = self.getClientsAsClientData()

        ui_speaker, ui_listener = self.getClientsAsUi(0)
        get_btn = lambda name: ui_listener.testWin.findChildren(
                QtGui.QPushButton, name)[0]

        play_btn = get_btn("btnPlay")
        submit_btn = get_btn("btnSubmit")
        # record_btn = get_btn("btnRecord")
        choices = [get_btn("btnImage%d" % i) for i in range(1, 5)]
        self.click(play_btn)
        self.click(choices[answer])
        print "Chosen the answer..."

        def submit():
            self.assertTrue(submit_btn.isEnabled())
            print "Clicking submit button, which is *enabled*"
            self.click(submit_btn)
            callback("FirstAnswer")

        self.reactor.callLater(1, submit)

    


class ServerTest(P2PTestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.stopServer()

    def setUp(self):
        prep(self)

    def test_startUp(self):
        self.startServer()
        self.assertIsNotNone(self.server_factory)
        self.assertIsNotNone(self.server_factory.ui)

    def test_invalidCondition(self):
        self.assertRaises(
            Exception, lambda: startServer(self.app, condition=1,
                                           no_ui=False))

    def test_headlessStartup(self):
        self.server_factory = start_server(
            self.app, condition='1', no_ui=True)
        self.assertIsNotNone(self.server_factory)
        self.assertIsNone(self.server_factory.ui)


class ServerTestWithClient(P2PTestCase):

    def tearDown(self):
        self.stopClients()
        self.stopServer()

    def setUp(self):
        prep(self)
        self.startServer()

    def test_connect(self):
        # client = start_client(self.app, 'client1')
        data = self.startClient(1).namedtuple
        print "Data:", data
        print "Connection:", data.connection

        d = defer.Deferred()
        def fn(client):
            # data.factory.ui.setClient(client)
            for factory in self.factories:
                if isinstance(factory, LeapP2PServerFactory):
                    item = "%s (%s)" % (data.client_ip, data.client_id)
                    print "Looking for: ", item
                    item = factory.ui.clientModel.findItems(item, Qt.MatchExactly)
                    print "Rows: ", factory.ui.clientModel.rowCount()
                    print "Found: ", item
                    d.callback(self.assertIsNotNone(item))

        # data.theremin.callLater(5, fn)
        data.connection.addCallback(fn)
        return data.connection
