__author__ = 'kerem'

from twisted.trial import unittest
from leaparticulator.data.functions import fromFile
import os


class TestFromFile(unittest.TestCase):

    def setUp(self):
        self.filenames = map(lambda x: os.path.join('logs', x), ["FL1.1.exp.log", "FL3.1.exp.log"])

    def tearDown(self):
        pass

    def testFL1(self):
        fromFile(self.filenames[0])

    def testFL2(self):
        fromFile(self.filenames[1])
