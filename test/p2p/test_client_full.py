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
    def test_createFirstSignal(self):
        self.click(self.factory.ui.mainWin.btnStart)
        d = defer.Deferred()
        def fn():
            speaker = False
            listener = False
            for client in self.clients:
                if client.factory.mode == Constants.SPEAKER:
                    print "Speaker"
                    speaker = client
                else:
                    print "Listener"
                    listener = client
            ui_speaker = speaker.factory.ui
            ui_listener = listener.factory.ui
            win_speaker = ui_speaker.creationWin
            get_btn = lambda name: win_speaker.findChildren(QtGui.QPushButton, name)[0]

            record_btn = get_btn("btnRecord")
            play_btn = get_btn("btnPlay")
            submit_btn = get_btn("btnSubmit")
            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]

            # submit and play start off disabled
            self.assertFalse(play_btn.isEnabled())
            self.assertFalse(submit_btn.isEnabled())

            # record something
            from LeapFrame import generateRandomSignal
            self.click(record_btn); self.click(record_btn)
            ui_speaker.theremin.last_signal = generateRandomSignal(10)

            # now things should be enabled
            self.assertTrue(play_btn.isEnabled())
            self.assertTrue(submit_btn.isEnabled())

            self.click(submit_btn)
            self.assertTrue(ui_speaker.is_waiting())
            d.callback(("FirstSignal"))
        self.reactor.callLater(.1, fn)
        
        return d
#     def test_answerFirstQuestion(self):
#         pass