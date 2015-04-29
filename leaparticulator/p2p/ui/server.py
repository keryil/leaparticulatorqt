#!/usr/bin/python
# -*- coding: utf-8 -*-

# simple.py
import sys

from PyQt4.QtGui import *


app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print "LeapP2PServerUI new QApp: %s" % app
else:
    print "LeapP2PServerUI existing QApp: %s" % app

from PyQt4.QtCore import *

from LeapTheremin import ThereminPlayback
from leaparticulator import constants
from QtUtils import loadUiWidget

def fn(self, event):
        event.ignore()

class LeapP2PServerUI(object):
    def __init__(self, app):
        self.app = app
        self.mainWin = None
        self.flickerDelta = 1
        self.session = None
        self.mainWin = loadUiWidget('ServerWindow.ui')

        self.lstParticipants = self.mainWin.findChildren(QListView, "lstParticipants")[0]
        self.clientModel = QStandardItemModel(self.lstParticipants)
        self.lstParticipants.setModel(self.clientModel)
        
        self.lstRounds = self.mainWin.findChildren(QListView, "lstRounds")[0]
        self.roundModel = QStandardItemModel(self.lstRounds)
        self.lstRounds.setModel(self.roundModel)

        self.lstRounds.selectionModel().selectionChanged.connect(self.displayRoundData)

        self.lblExpected = self.mainWin.findChildren(QLabel, "lblExpected")[0]
        self.lblGiven = self.mainWin.findChildren(QLabel, "lblGiven")[0]
        
        self.lblCondition = self.mainWin.findChildren(QLabel, "lblCondition")[0]
        self.lblSpeaker = self.mainWin.findChildren(QLabel, "lblSpeaker")[0]
        self.lblHearer = self.mainWin.findChildren(QLabel, "lblHearer")[0]
        self.playback = ThereminPlayback(default_volume = None)
        self.btnPlay = self.mainWin.findChildren(QPushButton, "btnPlay")[0]
        self.btnStart = self.mainWin.findChildren(QPushButton, "btnStart")[0]
        # self.go()

    def get_active_window(self):
        for w in (self.mainWin, ):
            if w:
                return w
        else:
            return None

    def close_all(self):
        w = self.get_active_window()
        if w:
            w.close()

    def enableStart(self):
        self.btnStart.setEnabled(True)

    def disableStart(self):
        self.btnStart.setEnabled(False)

    def connectionMade(self, client, client_id):
        print "Connection made with %s!" % client_id
        item = QStandardItem("%s (%s)" % (client, client_id))
        self.clientModel.appendRow(item)

    def connectionLost(self, client, reason):
        # print "Connection lost, the UI knows!"
        client_id = client.other_end_alias
        client_ip = client.other_end
        item = "%s (%s)" % (client_ip, client_id)
        item = self.clientModel.findItems(item, Qt.MatchExactly)
        # this prevents a race condition
        # where if the connect/disconnect
        # is too fast, there might never be
        # a QStandardItem added.
        if item != []:
            item = item[0].row()
            self.clientModel.removeRow(item)
        
    def flicker(self):
        """
        This method simply jiggles the mouse. It is used as 
        part of a workaround for LoopingCall's getting stuck
        on the first call when using qt4reactor. 
        """
        w = self.get_active_window()
        if w:
            from pymouse import PyMouse
            m = PyMouse()
            x,y = m.position()
            m.move(x+self.flickerDelta,y)
            self.flickerDelta *= -1

    def onSessionChange(self, session):
        self.session = session
        self.lblCondition.setText("Condition: %s" % session.condition)
        for index, rnd in enumerate(session.round_data):
            title = "Round #%s" % index
            items = self.roundModel.findItems(title)
            if items != []:
                item = items[0]
            else:
                item = QStandardItem(title)
                self.roundModel.appendRow(item)

    def displayRoundData(self, old, new):
        row = self.lstRounds.selectedIndexes()[0].row()
        rnd = self.session.round_data[row]
        print "Selected round #%d" % row
        print "Speaker: %s, Hearer: %s\nImage: %s, Guess: %s" % (rnd.speaker,
                                                                 rnd.hearer,
                                                                 rnd.image,
                                                                 rnd.guess)
        self.lblSpeaker.setText("Speaker: %s" % rnd.speaker.other_end_alias)
        self.lblHearer.setText("Hearer: %s" % rnd.hearer.other_end_alias)
        # if rnd.signal and len(rnd.signal) > 5:
        #     print "Signal:", rnd.signal[:5]
        # else:
            # print "Signal:", rnd.signal
        if rnd.image != None:
            self.lblExpected.setPixmap(rnd.image.pixmap())
        else:
            self.lblExpected.setPixmap(QPixmap(constants.question_mark_path))
        if rnd.guess != None:
            self.lblGiven.setPixmap(rnd.guess.pixmap())
        else:
            self.lblGiven.setPixmap(QPixmap(constants.question_mark_path))
        
        if rnd.signal != None and rnd.signal != []:
            def play_back():
                self.playback.start(rnd.signal)
            self.btnPlay.clicked.connect(play_back)


    def first_screen(self):
        # self.close_all()
        # button = self.mainWin.findChildren(QPushButton, "btnOkay")[0]
        self.mainWin.show()
        # button.clicked.connect(self.creation_screen)
        print("olee")
        return self.mainWin


    def show_wait(self, parent=None):
        if not parent:
            parent = self.get_active_window()
        self.waitDialog = loadUiWidget('WaitDialog.ui', parent)
        # self.waitDialog.setParent(parent)#QDialog(parent)
        #flags = #Qt.WindowStaysOnTopHint# | Qt.WindowTitleHint 
        # flags = flags & ~Qt.WindowCloseButtonHint
                                       
        # self.waitDialog.setWindowFlags(flags)
        self.waitDialog.setModal(True)
        pos = QApplication.desktop().screen().rect().center()- self.waitDialog.rect().center()
        self.waitDialog.move(pos)
        self.waitDialog.show()
        self.waitDialog.activateWindow()
        self.waitDialog.setFocus()
        # self.waitDialog.closeEvent = fn

    def wait_over(self):
        self.waitDialog.close()

    def go(self):

        # Create Qt application and the QDeclarative view
        # self.app = QApplication.instance()
        # self.app = app
        # if not self.app:
        #     self.app = QApplication(sys.argv)
        # mainWin = loadUiWidget('qt_interface/mainWindow.ui')
        # button = mainWin.findChildren(QPushButton, "btnOkay")[0]
        # button.clicked.connect(mainWin.close)
        # mainWin.showFullScreen()
        self.first_screen()
        # self.creation_screen(QPixmap("img/meanings/5_5.png"))
        # show_wait()
        # view = QDeclarativeView()
        # # Create an URL to the QML file
        # url = QUrl('qt_interface/P2PLeapArticulator/P2PLeapArticulator.qml')
        # # Set the QML file and show
        # view.setSource(url)
        # Enter Qt main loop

if __name__ == "__main__":
    gui = LeapP2PServerUI(app)
    print "About to go..."
    gui.go()
    sys.exit(app.exec_())