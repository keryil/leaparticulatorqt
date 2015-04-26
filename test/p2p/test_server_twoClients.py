import sys
from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
from PyQt4 import QtGui

from LeapP2PServer import start_server, start_client
from test_server_basic import prep, P2PTestCase
import Constants
from twisted.internet import defer


class TwoClientsInit(P2PTestCase):

    def tearDown(self):
        self.stopServer()
        self.clients = []

    def setUp(self):
        from twisted.internet import reactor
        self.reactor = reactor
        prep(self)
        self.timeout = 7
        self.startServer()

    def test_connect(self):
        self.startClient(1)
        button = self.factory.ui.mainWin.btnStart
        print "Button enabled? ",  button.isEnabled()
        self.assertFalse(button.isEnabled())
        self.startClient(2)

    def test_enableStart(self):
        self.startClients(2)
        # self.reactor.iterate(1)
        d = defer.Deferred()
        button = self.factory.ui.mainWin.btnStart

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
            lst = [self.assertTrue(client.factory.ui.is_waiting()) for client in self.clients]
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

        self.clients = self.startClients(2)
        d = defer.Deferred()
        def clickOkay():
            for client in self.clients:
                button = client.factory.ui.firstWin.findChildren(
                    QtGui.QPushButton, "btnOkay")[0]
                self.click(button)
            def clickStart():
                self.click(self.factory.ui.mainWin.btnStart)
                # dd = defer.Deferred()
                self.reactor.callLater(.3, lambda : d.callback("setUp"))
                # return dd
            self.reactor.callLater(.3, clickStart)
        self.reactor.callLater(.3, clickOkay)
        return d

    def test_roundList(self):
        items = self.factory.ui.roundModel.findItems("Round #0")
        self.failIfEqual(items, [])

    def test_roundDisplayForSpeaker(self):
        speaker, listener = self.getClients()

        view = self.factory.ui.lstRounds
        index = view.model().index(0,0)
        view.setCurrentIndex(index)
        self.click(view)
        d = defer.Deferred()
        def test():
            client_id = str(self.factory.ui.lblSpeaker.text()).split()[-1]
            self.assertEqual(speaker.client_id, client_id)
            
            client_id = str(self.factory.ui.lblHearer.text()).split()[-1]
            self.assertEqual(listener.client_id, client_id)

            speaker_img = speaker.factory.current_speaker_image

            # compare speaker's original image and the one displayed 
            self.assertEqual(speaker_img.pixmap().toImage(), 
                             self.factory.ui.lblExpected.pixmap().toImage())
            self.assertEqual(self.factory.ui.lblGiven.pixmap().toImage(), 
                             QtGui.QPixmap(Constants.question_mark_path).toImage())
            d.callback(client_id)
        self.reactor.callLater(.2, test)
        return d

    def test_askFirstQuestion(self):
        pass

#     def test_endOfFirstRoundServerUI(self):
#         pass
