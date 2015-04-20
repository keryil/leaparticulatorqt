import sys
from twisted.trial import unittest
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from LeapP2PServer import start_server, start_client
from twisted.internet import defer


def prep(self):
    import Constants
    Constants.leap_server = "127.0.0.1"
    self.app = QApplication.instance()
    if not self.app:
        self.app = QApplication(sys.argv)


class P2PServerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'factory'):
            self.factory.listener.result.stopListening()
            self.factory.stopFactory()
        # self.app.quit()
        from time import sleep
        sleep(1)

    def setUp(self):
        prep(self)

    def test_startUp(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=False)
        self.assertIsNotNone(self.factory)

    def test_invalidCondition(self):
        self.assertRaises(
            Exception, lambda: start_server(self.app, condition=1, no_ui=False))

    def test_headlessStartup(self):
        self.factory = start_server(
            self.app, condition='1', no_ui=True)
        self.assertIsNotNone(self.factory)
        self.assertIsNone(self.factory.ui)
