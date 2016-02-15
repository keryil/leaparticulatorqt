#!/usr/bin/python
# -*- coding: utf-8 -*-

# simple.py
import re
import sys

from PyQt4.QtGui import *

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print "LeapP2PServerUI new QApp: %s" % app
else:
    print "LeapP2PServerUI existing QApp: %s" % app

from PyQt4.QtCore import *

# from LeapTheremin import ThereminPlayback
from leaparticulator.theremin.theremin import ThereminPlayback
from leaparticulator import constants
from leaparticulator.oldstuff.QtUtils import loadUiWidget


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
        self.playback = ThereminPlayback(default_volume=None)
        self.btnPlay = self.mainWin.findChildren(QPushButton, "btnPlay")[0]
        self.btnStart = self.mainWin.findChildren(QPushButton, "btnStart")[0]
        self.btnEnd = self.mainWin.findChildren(QPushButton, "btnEnd")[0]

        self.meaningSpaceScroll = self.mainWin.findChildren(QScrollArea, "scrollArea")
        self.meaningSpace = self.mainWin.findChildren(QGroupBox, "meaningSpace")[0]
        self.meaningSpaceLabels = []
        # self.meaningSpace = self.mainWin.findChildren(QGroupBox, "meaningSpace")[0]
        # self.meaningSpaceLayout = self.mainWin.findChildren(QVBoxLayout, "meaningLayout")[0]
        # self.go()

    def get_active_window(self):
        for w in (self.mainWin,):
            if w:
                return w
        else:
            return None

    def close_all(self):
        w = self.get_active_window()
        if w:
            w.close()

    def setFactory(self, factory):
        self.factory = factory

        def end():
            for client in self.factory.clients:
                client.end()

        self.btnEnd.clicked.connect(end)

    def enableStart(self):
        self.btnStart.setEnabled(True)

    def disableStart(self):
        self.btnStart.setEnabled(False)

    def enableEnd(self):
        self.btnEnd.setEnabled(True)

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
            x, y = m.position()
            m.move(x + self.flickerDelta, y)
            self.flickerDelta *= -1

    def onSessionChange(self, session):
        self.session = session
        self.lblCondition.setText("Condition: %s" % session.condition)
        phase = -1
        last_pointer = -1
        for index, rnd in enumerate(session.round_data):
            if rnd.image_pointer > last_pointer:
                phase += 1
                last_pointer = rnd.image_pointer
            title = "Round #{} (Phase {})".format(str(index).zfill(2),
                                                  phase)
            items = self.roundModel.findItems(title)
            if items != []:
                item = items[0]
            else:
                item = QStandardItem(title)
                self.roundModel.appendRow(item)
        self.displayRoundData(implicit_update=True)

    def displayRoundData(self, implicit_update=False):
        # if there is nothing selected, do nothing
        if implicit_update and (self.lstRounds.selectedIndexes() == []):
            return
        row = self.lstRounds.selectedIndexes()[0].row()
        rnd = self.session.round_data[row]

        if not implicit_update:
            print "Selected round #%d" % row
            print "Speaker: %s, Hearer: %s\nImage: %s, Guess: %s" % (rnd.speaker.other_end_alias,
                                                                 rnd.hearer.other_end_alias,
                                                                 rnd.image,
                                                                 rnd.guess)
        self.lblSpeaker.setText("Speaker: %s" % rnd.speaker.other_end_alias)
        self.lblHearer.setText("Hearer: %s" % rnd.hearer.other_end_alias)
        # if rnd.signal and len(rnd.signal) > 5:
        #     print "Signal:", rnd.signal[:5]
        # else:
        # print "Signal:", rnd.signal
        if rnd.image is not None:
            overlay = ''.join(str(rnd.image.feature_dict[feature])
                              for feature in rnd.image.feature_order)
            self.lblExpected.setPixmap(rnd.image.pixmap(overlayed_text=overlay,
                                                        font=QFont("Arial", 25)))
            self.lblExpected.meaning = rnd.image
        else:
            self.lblExpected.setPixmap(QPixmap(constants.question_mark_path))

        if rnd.guess is not None:
            overlay = ''.join(str(rnd.guess.feature_dict[feature])
                              for feature in rnd.guess.feature_order)
            self.lblGiven.setPixmap(rnd.guess.pixmap(overlayed_text=overlay,
                                                     font=QFont("Arial", 25)))
            self.lblGiven.meaning = rnd.guess
        else:
            self.lblGiven.setPixmap(QPixmap(constants.question_mark_path))

        if rnd.signal:
            def play_back():
                def stop():
                    self.btnPlay.setText("Play the signal")
                    self.btnPlay.clicked.disconnect()
                    self.btnPlay.clicked.connect(play_back)

                self.btnPlay.setText("Stop playback")
                self.btnPlay.clicked.disconnect()
                self.btnPlay.clicked.connect(stop)
                self.playback.start(rnd.signal, callback=stop)

            self.btnPlay.clicked.connect(play_back)
        self.displayMeaningSpace(rnd)

    def displayMeaningSpace(self, rnd):
        """
        Displays the meaning space and the associated counts on
        the server GUI.
        :param rnd:
        :return:
        """
        pointer = rnd.image_pointer
        print "Image pointer with which I'll draw the meaning space: %s" % pointer
        image_list = self.factory.images[:pointer]
        if not hasattr(self, 'meaningSpace'):
            print "Creating the initial meaning space widgets..."
            self.meaningSpace = QGroupBox(title="Meaning Space")
            self.meaningSpace.setObjectName("meaningSpace")
            self.mainWin.findChildren(QScrollArea, "scrollArea")[0].setWidget(self.meaningSpace)

        assert self.meaningSpace
        print "Got a handle for the meaning space now."

        # first clean the current items, if present
        layout = self.meaningSpace.layout()
        if layout:
            # print "Cleaning old items in the meaning space..."
            for labels in self.meaningSpaceLabels:
                for label in labels:
                    layout.removeWidget(label)
                    QObjectCleanupHandler().add(label)
            self.meaningSpaceLabels = []
        else:
            layout = QFormLayout()
            self.meaningSpace.setLayout(layout)

        for meaning in map(str, image_list):
            count = rnd.success_counts[meaning]
            # print "Adding %s to meaning space" % meaning
            index = map(str, image_list).index(meaning)
            label = QLabel()
            overlay = re.search('(\d+)\.png', meaning).group(1)
            label.setPixmap(image_list[index].pixmap(overlayed_text=overlay,
                                                     font=QFont("Arial", 35)).
                            scaledToWidth(75, mode=Qt.SmoothTransformation))
            count_label = QLabel("%s correct guesses" % count)
            layout.addRow(label, count_label)
            self.meaningSpaceLabels.append([label, count_label])
            # print meaning, count

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
        # flags = #Qt.WindowStaysOnTopHint# | Qt.WindowTitleHint
        # flags = flags & ~Qt.WindowCloseButtonHint

        # self.waitDialog.setWindowFlags(flags)
        self.waitDialog.setModal(True)
        pos = QApplication.desktop().screen().rect().center() - self.waitDialog.rect().center()
        self.waitDialog.move(pos)
        self.waitDialog.show()
        self.waitDialog.activateWindow()
        self.waitDialog.setFocus()
        # self.waitDialog.closeEvent = fn

    def wait_over(self):
        self.waitDialog.close()

    def go(self):
        self.first_screen()


if __name__ == "__main__":
    gui = LeapP2PServerUI(app)
    print "About to go..."
    gui.go()
    sys.exit(app.exec_())
