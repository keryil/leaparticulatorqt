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

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.factory.listener.result.stopListening()
            self.factory.stopFactory()
        # self.app.quit()

    def setUp(self):
        prep(self)
        self.factory = start_server(
            self.app, condition='1', no_ui=False)

    def test_startUp(self):
        self.client = start_client(
            self.app, uid="test1")
        self.assertIsNotNone(self.client)

    # def test_invalidCondition(self):
    #     self.assertRaises(
    #         Exception, lambda: start_server(self.app, condition=1, no_ui=False))

    # def test_headlessStartup(self):
    #     self.factory = start_server(
    #         self.app, condition='1', no_ui=True)
    #     self.assertIsNotNone(self.factory)
    #     self.assertIsNone(self.factory.ui)
