from datetime import datetime
import sys

import qt4reactor
from PySide import QtGui, QtCore
from PySide.QtGui import QApplication


qapplication = QApplication(sys.argv)
qt4reactor.install()
from twisted.internet.task import LoopingCall
from twisted.internet import reactor


class LCDRange(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        lcd = QtGui.QLCDNumber(2)
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 99)
        slider.setValue(0)
        self.connect(slider, QtCore.SIGNAL("valueChanged(int)"),
                     lcd, QtCore.SLOT("display(int)"))

        layout = QtGui.QVBoxLayout()
        layout.addWidget(lcd)
        layout.addWidget(slider)
        self.setLayout(layout)


class MyWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        quit = QtGui.QPushButton("Quit")
        quit.setFont(QtGui.QFont("Times", 18, QtGui.QFont.Bold))
        self.connect(quit, QtCore.SIGNAL("clicked()"),
                     QtGui.qApp, QtCore.SLOT("quit()"))

        grid = QtGui.QGridLayout()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(quit)
        layout.addLayout(grid)
        self.setLayout(layout)
        for row in range(3):
            for column in range(3):
                grid.addWidget(LCDRange(), row, column)


def hyper_task():
    print "I like to run fast", datetime.now()


def tired_task():
    print "I want to run slowly", datetime.now()


lc = LoopingCall(hyper_task)
lc.start(0.1)

lc2 = LoopingCall(tired_task)
lc2.start(0.5)
w = MyWidget()
w.show()
reactor.runReturn()
sys.exit(qapplication.exec_())