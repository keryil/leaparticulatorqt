import sys
from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from LeapP2PServer import start_server, start_client
from test_server_basic import prep
import Constants
from twisted.internet import defer


class P2PClientTest(unittest.TestCase):

    # def tearDown(self):
    #     if hasattr(self, 'factory'):
    #         self.factory.listener.result.stopListening()
    #         self.factory.stopFactory()
        # self.app.quit()

    def setUp(self):
        prep(self)
        # self.factory = start_server(
            # self.app, condition='1', no_ui=False)

    def test_startUp(self):
        theremin, reactor, controller, connection, factory = start_client(
            self.app, uid="test1")
        self.assertIsNotNone(theremin)
        self.assertIsNotNone(controller)
        self.assertIsNotNone(factory)

    def test_noUID(self):
        self.assertRaises(Exception, lambda : start_client(
            self.app, uid=None))


class P2PClientTestWithServer(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.factory.listener.result.stopListening()
            self.factory.stopFactory()

    def setUp(self):
        prep(self)
        self.factory = start_server(
            self.app, condition='1', no_ui=False)

    def test_connect(self):
        theremin, reactor, controller, connection, factory = start_client(
            self.app, uid="test1", run=True)
        factory.ui.go()
        return factory.connection_def
