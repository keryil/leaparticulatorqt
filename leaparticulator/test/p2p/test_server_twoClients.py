from PyQt4 import QtGui
from twisted.internet import defer

from test_server_basic import prep, P2PTestCase


class TwoClientsInit(P2PTestCase):

    def tearDown(self):
        self.stopServer()
        self.clients = {}

    def setUp(self):
        from twisted.internet import reactor
        self.reactor = reactor
        prep(self)
        self.timeout = 7
        self.startServer()

    def test_connect(self):
        self.startClient(1)
        button = self.server_factory.ui.mainWin.btnStart
        print "Button enabled? ",  button.isEnabled()
        self.assertFalse(button.isEnabled())
        self.startClient(2)

    def test_enableStart(self):
        self.startClients(2)
        # self.reactor.iterate(1)
        d = defer.Deferred()
        button = self.server_factory.ui.mainWin.btnStart

        def fn():
            print "Button enabled? ",  button.isEnabled()
            d.callback(self.assertTrue(button.isEnabled()))
        self.reactor.callLater(.2, fn)
        return d

    def test_OkayButtonsBeforeStart(self):
        self.clients = self.startClients(2)
        # self.reactor.iterate(1)
        # self.factory.ui.mainWin.btnStart
        d = defer.Deferred()
        # d2 = defer.Deferred()
        # d1.chainDeferred(d2)
        for client in self.clients:
            button = client.factory.ui.firstWin.findChildren(
                QtGui.QPushButton, "btnOkay")[0]
            self.click(button)

        def fn():
            lst = [self.assertTrue(client.factory.ui.is_waiting())
                   for client in self.clients]
            d.callback(lst)
        self.reactor.callLater(.1, fn)
        return d


class TwoClientsFirstRound(P2PTestCase):

    def tearDown(self):
        self.stopServer()
        self.clients = []

    def setUp(self):
        from twisted.internet import reactor
        self.reactor = reactor
        prep(self)
        self.startServer()
        self.timeout = 3

        self.clients = {c.client_id:c for c in self.startClients(2)}
        d = defer.Deferred()

        def clickOkay():
            for client_id, client in self.clients.items():
                button = client.factory.ui.firstWin.findChildren(
                    QtGui.QPushButton, "btnOkay")[0]
                self.click(button)

            def clickStart():
                self.click(self.server_factory.ui.mainWin.btnStart)
                # dd = defer.Deferred()
                self.reactor.callLater(.3, lambda: d.callback("setUp"))
                # return dd
            self.reactor.callLater(.3, clickStart)
        self.reactor.callLater(.3, clickOkay)
        return d

    def test_roundList(self):
        items = self.server_factory.ui.roundModel.findItems("Round #0")
        self.failIfEqual(items, [])

    def test_roundSpeakerHearerDisplay(self):
        speaker, listener = self.getClientsAsServerConnections(rnd_no=0)

        view = self.server_factory.ui.lstRounds
        index = view.model().index(0, 0)
        view.setCurrentIndex(index)
        self.click(view)
        d = defer.Deferred()

        def test():
            client_id = str(self.server_factory.ui.lblSpeaker.text()).split()[-1]
            self.assertEqual(client_id, speaker.other_end_alias)

            client_id = str(self.server_factory.ui.lblHearer.text()).split()[-1]
            self.assertEqual(client_id, listener.other_end_alias)
            d.callback(client_id)
        self.reactor.callLater(.3, test)
        return d

    def test_roundSpeakerHearerImage(self):
        view = self.server_factory.ui.lstRounds
        index = view.model().index(0, 0)
        view.setCurrentIndex(index)
        self.click(view)
        d = defer.Deferred()

        def test():
            # from time import sleep;sleep(30)
            speaker_img = self.getRound(rnd_no=0).image

            # hearer's image should be displayed as a question mark
            self.assertFalse(hasattr(self.server_factory.ui.lblGiven, 'meaning'))
            # compare speaker's original image and the one displayed
            self.assertEqual(speaker_img,
                             self.server_factory.ui.lblExpected.meaning)
            d.callback("Images")
        self.reactor.callLater(.3, test)
        return d

    def test_askFirstQuestion(self):
        pass

#     def test_endOfFirstRoundServerUI(self):
#         pass
