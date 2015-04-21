import sys
from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
import PyQt4

from LeapP2PServer import start_server, start_client
from test_server_basic import prep, P2PTestCase
import Constants
from twisted.internet import defer



class P2PServerTestWithClient(P2PTestCase):

    def tearDown(self):
        self.stopServer()

    def setUp(self):
        prep(self)
        self.startServer()

    def test_connect(self):
        self.startClient(1)
        button = self.factory.ui.mainWin.btnStart
        # print "Button enabled? ",  button.isEnabled()
        self.assertFalse(button.isEnabled())
        self.startClient(2)

    def test_enableStart(self):
        for id in range(2):
            self.startClient(id)
        self.reactor.iterate(1)
        button = self.factory.ui.mainWin.btnStart
        # print "Button enabled? ",  button.isEnabled()
        self.assertTrue(button.isEnabled())