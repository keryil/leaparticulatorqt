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
        self.reactor.callLater(.1, fn)
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
        for client in self.clients:
            button = client.factory.ui.firstWin.findChildren(
                QtGui.QPushButton, "btnOkay")[0]
            self.click(button)
        d = defer.Deferred()
        self.reactor.callLater(.2, lambda : d.callback('setUp'))
        return d


    def test_FirstImage(self):
        self.click(self.factory.ui.mainWin.btnStart)
        d = defer.Deferred()
        def fn():
            print self.factory.mode
            self.assertEqual(self.factory.mode, Constants.SPEAKERS_TURN)
            speaker = False
            listener = False
            for client in self.clients:
                if client.factory.mode == Constants.SPEAKER:
                    print "Speaker: ", client
                    speaker = client
                else:
                    print "Listener: ", client
                    listener = client
            ui_speaker = speaker.factory.ui
            ui_listener = listener.factory.ui
            
            self.assertTrue(speaker)
            self.assertEqual(speaker.factory.mode, Constants.SPEAKER)
            self.assertTrue(listener)
            self.assertEqual(listener.factory.mode, Constants.LISTENER)
            self.assertEqual(
                ui_speaker.get_active_window(), ui_speaker.creationWin)
            self.assertEqual(ui_listener.get_active_window(), ui_listener.firstWin)
            self.assertTrue(ui_listener.is_waiting())
            self.assertFalse(ui_speaker.is_waiting())

            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]
            self.assertEqual(
                speaker.factory.current_image.pixmap().toImage(), image.pixmap().toImage())
            d.callback('FirstImage')
        self.reactor.callLater(.1, fn)
        return d




#     def test_askFirstQuestion(self):
#         pass

#     def test_endOfFirstRoundServerUI(self):
#         pass
