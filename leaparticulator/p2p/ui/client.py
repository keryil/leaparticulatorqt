#!/usr/bin/python
# -*- coding: utf-8 -*-

# simple.py
import sys

from PyQt4.QtGui import *
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print "LeapP2PClient new QApp: %s" % app
else:
    print "LeapP2PClient existing QApp: %s" % app

from leaparticulator.theremin.theremin import ThereminPlayback
# from LeapTheremin import ThereminPlayback
from leaparticulator.p2p.messaging import EndRoundMessage
from QtUtils import connect, disconnect, loadUiWidget
import leaparticulator.constants as constants
# from QtUtils import loadWidget as loadUiWidget

# def loadUiWidget(name, parent=None):
#     widget = uic.loadUi("qt_generated/%s" % name)
#     widget.setParent(parent)
#     return widget
     
def fn(self, event):
        event.ignore()

class LeapP2PClientUI(object):
    def __init__(self, app, factory=None):
        self.slots_dict = {}
        self.factory = factory
        if factory:
            self.theremin = factory.theremin
        self.app = app
        self.firstWin = None
        self.creationWin = None
        self.testWin = None
        self.feedbackWin = None
        self.waitDialog = None
        self.flickerDelta = 1
        self.playback_player = ThereminPlayback(default_volume=None)
        self.last_signal = []
        self.recording = False
        self.finalScreen = None

    def setClientFactory(self, factory):
        self.factory = factory
        self.theremin = factory.theremin

    def setClient(self, client):
        self.client = client
        self.setClientFactory(client.factory)

    def get_active_window(self):
        for w in (self.firstWin, self.creationWin, 
                  self.testWin, self.feedbackWin):
            if w and w.isVisible():
                return w
        else:
            return None

    def close_all(self):
        for w in (self.firstWin, self.creationWin, 
                  self.testWin, self.feedbackWin, 
                  self.waitDialog):
            if w and w.isVisible():
                w.close()

    def flicker(self):
        """
        This method simply jiggles the mouse. It is used as 
        part of a workaround for LoopingCall's getting stuck
        on the first call when using qt4reactor. 
        """
        w = self.get_active_window()
        from platform import system
        if w and system() == "Linux":
            from pymouse import PyMouse
            m = PyMouse()
            x,y = m.position()
            m.move(x+self.flickerDelta,y)
            self.flickerDelta *= -1

    def first_screen(self):
        self.close_all()
        # import pdb;pdb.set_trace()
        self.firstWin = loadUiWidget('FirstWindow.ui')
        button = self.firstWin.findChildren(QPushButton, "btnOkay")[0]
        text = self.firstWin.findChildren(QTextBrowser, "textBrowser")[0]
        print "loaded"
        self.theremin.unmute()
        self.firstWin.showFullScreen()
        connect(button, "clicked()", self.show_wait)
        from os.path import join
        text.setText(open(join(constants.P2P_RES_DIR, "first_screen.txt")).read())
        print "first_screen done"
        return self.firstWin

    def unique_connect(self, widget, signal, slot):
        disconnect(widget)
        connect(widget, signal, slot)

    def creation_screen(self,image=None):
        # close previous windows
        self.close_all()

        self.creationWin = loadUiWidget('SignalCreation.ui')

        def submit_and_proceed():
            self.client.speak()
            self.show_wait()

        button = self.creationWin.findChildren(QPushButton, "btnSubmit")[0]
        connect(button, "clicked()", submit_and_proceed)
        button.setEnabled(False)

        button = self.creationWin.findChildren(QPushButton, "btnRecord")[0]
        self.unique_connect(button, "clicked()", self.start_recording)

        button = self.creationWin.findChildren(QPushButton, "btnPlay")[0]
        button.setEnabled(False)
        # self.setup_play_button(button, self.getSignal())

        label = self.creationWin.findChildren(QLabel, "lblImage")[0]
        px = image.pixmap()
        print "Displaying pixmap: %s" % px.toImage()
        label.setPixmap(px)

        slider = self.creationWin.findChildren(QSlider, "sldVolume")[0]
        slider.setRange(1, 100)
        slider.setSingleStep(1)
        slider.setValue(100)
        self.unique_connect(slider, "valueChanged(int)", self.set_volume)
        # slider.valueChanged.connect(self.set_volume)

        # self.show_wait(self.creationWin)
        self.creationWin.showFullScreen()
        return self.creationWin

    def set_volume(self, value):
        value = value / 100.
        self.playback_player.setVolume(value)
        self.theremin.setVolume(value)

    def start_recording(self):
        if self.creationWin and not self.recording:
            print "Start recording..."
            self.theremin.reset_signal()
            self.theremin.unmute()
            self.theremin.record()
            self.recording = True
            # self.factory.start_recording()
            button = self.creationWin.findChildren(QPushButton, "btnRecord")[0]
            button.setText("Stop")
            self.unique_connect(button, "clicked()", self.stop_recording)
            self.flicker()


    def setup_play_button(self, button, signal):
        # this is a flag which is only true if the associated
        # audio has been played at least once.
        self.played = False

        def enable(button=button):
            # button = self.creationWin.findChildren(QPushButton, "btnPlay")[0]
            self.played = True
            button.setEnabled(True)
            self.unique_connect(button, "clicked()", play)
            if (self.get_active_window() == self.testWin) and self.picked_choice:
                button = self.testWin.findChildren(QPushButton, "btnSubmit")[0]
                button.setEnabled(True)

        def play():
            button.setEnabled(False)
            # button.setText("Stop")
            self.unique_connect(button, "clicked()", self.playback_player.stop)
            self.playback_player.start(signal, 
                                       enable)
            self.flicker()
        
        self.unique_connect(button, "clicked()", play)

    def stop_recording(self):
        if self.creationWin and self.recording:
            # self.factory.stop_recording()
            self.recording = False
            self.theremin.stop_record()
            # self.theremin.mute()
            btnRec = self.creationWin.findChildren(QPushButton, "btnRecord")[0]
            btnRec.setText("Re-record")
            self.unique_connect(btnRec, "clicked()", self.start_recording)

            btnPlay = self.creationWin.findChildren(QPushButton, "btnPlay")[0]
            btnPlay.setEnabled(True)

            btnSubmit = self.creationWin.findChildren(QPushButton, "btnSubmit")[0]
            btnSubmit.setEnabled(True)

            self.setup_play_button(btnPlay, self.getSignal())
            self.flicker()
            print "Signal is %d frames long." % len(self.getSignal())

    def extend_last_signal(self, frame):
        if self.recording:
            # self.last_signal.append(frame)
            self.theremin.last_signal.append(frame)
            # print ("Extending signal at ui... (frame %d)" % len(self.getSignal()))
            
    def resetSignal(self):
        self.theremin.reset_signal()

    def getSignal(self):
        return self.theremin.get_signal()

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
        parent.setEnabled(False)
        # self.waitDialog.closeEvent = fn

    def wait_over(self):
        if self.waitDialog:
            self.waitDialog.parent().setEnabled(True)
            self.waitDialog.close()
            self.waitDialog = None

    def is_waiting(self):
        return self.waitDialog and self.waitDialog.isVisible()

    def test_screen(self,images=None):
        # close previous windows
        self.close_all()
        self.wait_over()
        # this is a flag which is only true if one of the options is 
        # chosen
        self.picked_choice = False
        def submit():
            image_ = None
            for i, image in zip(range(1,5), images):
                button = self.testWin.findChildren(QPushButton, "btnImage%d" % i)[0]
                if button.isChecked():
                    image_ = image
                    break
            self.client.listen(image_)

        self.testWin = loadUiWidget('SignalTesting.ui')
        btnSubmit = self.testWin.findChildren(QPushButton, "btnSubmit")[0]
        connect(btnSubmit, "clicked()", submit)
        def enable():
            self.picked_choice = True
            if self.played:
                btnSubmit.setEnabled(True)

        button = self.testWin.findChildren(QPushButton, "btnPlay")[0]
        self.setup_play_button(button, self.factory.last_response_data.signal)
        
        slider = self.testWin.findChildren(QSlider, "sldVolume")[0]
        slider.setRange(1,100)
        slider.setSingleStep(1)
        slider.setValue(100)
        slider.valueChanged.connect(self.set_volume)

        for i in range(1, 5):
            view = self.testWin.findChildren(QPushButton, "btnImage%d" % i)[0]
            try:
                image = images[i - 1]
                pixmap = image.pixmap()
                # pixmap.setAlignment()
                view.setIcon(QIcon(pixmap))
                view.setIconSize(pixmap.rect().size())
                connect(view, "clicked()", enable)
            # this happens when we have fewer than 4 images
            except IndexError:
                view.setEnabled(False)


        self.testWin.showFullScreen()

        return self.testWin

    def feedback_screen(self, image_true, image_guess):
        # close previous windows

            # self.show_wait()

        self.close_all()

        self.feedbackWin = loadUiWidget('Feedback.ui')
        button = self.feedbackWin.findChildren(QPushButton, "btnOkay")[0]
        # if final:

        def proceed():
            self.client.send_to_server(EndRoundMessage())
            button.setEnabled(False)
            button.setText("Waiting for the other participant to click this button")

        connect(button, "clicked()", proceed)
        self.feedbackWin.findChildren(QLabel, "lblImageCorrect")[0].setPixmap(image_true.pixmap())
        self.feedbackWin.findChildren(QLabel, "lblImageGuess")[0].setPixmap(image_guess.pixmap())
        # else:
        #     button.clicked.connect(self.creation_screen)
        self.feedbackWin.showFullScreen()
        return self.feedbackWin

    def go(self):
        self.first_screen()

    def final_screen(self):
        from os.path import join
        self.close_all()
        self.finalScreen = loadUiWidget(constants.P2P_FINAL_WIN)
        self.finalScreen.btnOkay.clicked.connect(self.app.exit)
        msg = open(join(constants.P2P_RES_DIR, "final_screen.txt")).read()
        self.finalScreen.textBrowser.setText(msg)
        self.finalScreen.showFullScreen()

if __name__ == "__main__":
    gui = LeapP2PClientUI(app)
    gui.go()
    sys.exit(app.exec_())