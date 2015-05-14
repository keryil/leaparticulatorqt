import sys

from PyQt4 import QtCore, QtGui

from leaparticulator import constants


app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print "Existing QApplication instance:", app

# from PySide.QtCore import QFile
from AbstractClientUI import AbstractClientUI
# from LeapTheremin import gimmeSomeTheremin, ThereminPlayback
from QtUtils import connect, disconnect, loadWidget, loadFromRes, setButtonIcon
from jsonpickle import decode, encode
from leaparticulator.constants import install_reactor
install_reactor()
from twisted.internet import reactor

def getFunction(widget):
    """
    Shortcut for widget.findChildren(type,label)[0]
    """

    def get(type, label):
        try:
            return widget.findChildren(type, label)[0]
        except IndexError:
            print "No objects of type %s called '%s' were found for widget %s" % (type, label, widget)
            return None

    return get


class ClientUI(AbstractClientUI):
    def __init__(self, condition):
        # super(QObject, self).__init__()
        assert condition in ('1', '2', '1r', '2r')
        self.condition = condition
        self.n_of_test_questions = 4

        self.phase = -1
        self.isPractice = False

        # this is a dict of responses, in the form
        # dict[phase][image] = [frame1, frame2,...]
        self.responses = {phase: {} for phase in range(3)}
        self.responses_practice = {phase: {} for phase in range(3)}

        # this is a dict of test answers, in the form
        # dict[phase] = [TestQuestion1, TestQuestion2,...]
        self.test_results = {phase: [] for phase in range(3)}
        self.test_results_practice = {phase: [] for phase in range(3)}

        # theremin stuff
        from leaparticulator.theremin.theremin import ConstantRateTheremin, ThereminPlayback
        self.theremin = ConstantRateTheremin(
        # self.theremin, self.reactor, self.controller, self.connection = gimmeSomeTheremin(
            n_of_tones=1, default_volume=.5, ui=self, realtime=False)
        self.reactor = reactor
        self.connection = self.theremin.protocol
        self.playback_player = ThereminPlayback()
        self.default_volume = 0.5
        self.default_pitch = 440.
        self.last_signal = []
        self.muted = False
        self.volume = .5

        self.isRecording = False

        self.activeWindow = None
        self.learningWindow = loadWidget(constants.LEARNING_WIN)
        self.infoWindow = loadWidget(constants.INFO_WIN)
        self.testWindow = loadWidget(constants.TEST_WIN)

        # this is to map clicked() signals of each button
        # to the same function with different arguments
        self.buttonSignalMapper = None

    def setupAudio(self, window):
        """
        Sets up the play/record buttons for the given window, as well
        as the volume dial. Submit button is also affected by this
        setup (enabled/disabled), but its signal connection is made in the
        corresponding window setup method.
        """
        get = getFunction(window)
        # setup the volume dial
        dial = get(QtGui.QDial, 'volumeDial')
        dial.setValue(self.volume * 100)

        disconnect(dial)

        def volumeChange(x):
            self.theremin.setVolume(x / 100.)
            self.playback_player.setVolume(x / 100.)
            self.volume = x / 100.

        connect(dial, 'valueChanged(int)', volumeChange)

        # setup the audio playback and recording
        play = get(QtGui.QPushButton, 'btnPlay')
        record = get(QtGui.QPushButton, 'btnRecord')
        submit = get(QtGui.QPushButton, 'btnSubmit')

        # shortcuts
        def shortcuts():
            play.setShortcut(QtGui.QKeySequence.fromString("P"))
            if record:
                record.setShortcut(QtGui.QKeySequence.fromString("R"))
            submit.setShortcut(QtGui.QKeySequence.fromString("S"))

        # disable submit button until the first recording
        # or until an image is chosen for a signal
        submit.setEnabled(False)

        # handle play/stop plus label changes
        def fn_play():
            last_submit_state = None

            def fn_done():
                self.playback_player.stop()
                play.setText("Play")
                if record is not None:
                    record.setEnabled(True)
                submit.setEnabled(last_submit_state)
                disconnect(play)
                connect(play, "clicked()", fn_play)
                if not record:
                    submit.setEnabled(True)
                shortcuts()

            play.setText("Stop")
            disconnect(play)
            last_submit_state = submit.isEnabled()
            connect(play, "clicked()", fn_done)
            if record is not None:
                record.setEnabled(False)
            submit.setEnabled(False)
            shortcuts()
            self.playback_player.start(self.last_signal, callback=fn_done)

        connect(play, "clicked()", fn_play)

        # setup the recording if there is a record button
        if record is not None:
            def fn_record():
                def fn_done():
                    self.stop_recording()
                    # self.send(Constants.END_REC)
                    # self.isRecording = False
                    record.setText("Record")
                    play.setEnabled(True)
                    submit.setEnabled(True)
                    disconnect(record)
                    connect(record, "clicked()", fn_record)
                    shortcuts()

                self.start_recording()
                # self.send(Constants.START_REC)
                # self.last_signal = []
                # self.isRecording = True
                disconnect(record)
                connect(record, "clicked()", fn_done)
                play.setEnabled(False)
                submit.setEnabled(False)
                record.setText("Stop")
                shortcuts()

            connect(record, "clicked()", fn_record)

            # make sure playback is only available once a recording
            # is in place
            play.setEnabled(False)

    def setupWindow(self, window, meaning=None, mode=None):
        # if this is not an info window, setup the
        # play/record buttons.
        if window != self.infoWindow:
            self.setupAudio(window)

        # call the corresponding setup method
        if window == self.learningWindow:
            self.setupLearningWindow(meaning)
        elif window == self.testWindow:
            self.setupTestWindow()
        elif window == self.infoWindow:
            self.setupInfoWindow(mode)

    def setupLearningWindow(self, meaning):
        get = getFunction(self.learningWindow)

        # setup the target meaning (image)
        label = get(QtGui.QLabel, 'lblImage')
        label.setPixmap(meaning.pixmap())
        label.repaint()
        print "Set pixmap to %s" % label

        button = get(QtGui.QPushButton, "btnSubmit")
        disconnect(button)
        connect(button, "clicked()", self.next_question)
        self.unmute()

    def setupInfoWindow(self, mode=constants.MOD_PREPHASE):
        """
        Sets up the info window for whatever purpose, purposes
        are given with the parameter 'modes',
        """
        if mode is None:
            mode = constants.MOD_PREPHASE

        # set the info text
        if mode == constants.MOD_PREPHASE:
            res_name = "phase%d" % self.phase
        elif mode == constants.MOD_PRETEST:
            res_name = "test%d" % self.phase
        elif mode == constants.MOD_EXIT:
            res_name = "final"
        elif mode == constants.MOD_FIRSTSCREEN:
            res_name = "firstscreen"

        first_or_last = mode in (constants.MOD_EXIT, constants.MOD_FIRSTSCREEN)
        first_or_last_or_pretest = mode in (
            constants.MOD_EXIT, constants.MOD_FIRSTSCREEN, constants.MOD_PRETEST)

        if self.isPractice and not first_or_last:
            res_name += "_practice"

        info = loadFromRes(res_name)
        label = self.infoWindow.findChildren(QtGui.QLabel, "lblInfo")[0]
        if self.isPractice and not first_or_last:
            info = "PRACTICE ROUND: " + info
        label.setText(info)

        # set the meaning space image
        label = self.infoWindow.findChildren(
            QtGui.QLabel, "lblMeaningSpace")[0]
        label.pixmap = None
        dimension = "dimensions"
        if self.phase == 0:
            dimension = "dimension"

        if not first_or_last_or_pretest:
            from subprocess import Popen
            from shlex import split
            from os.path import join
            from os import remove

            images = self.images[self.phase]
            filename = join(constants.MEANING_DIR, "montage.png")
            command = "montage \"%s\" \"%s\"" % ("\" \"".join([i.filename() for i in images]),
                                                 filename)
            print "Montage creation command: %s" % command
            out, err = Popen(split(command)).communicate()
            print "Montage creation output: %s\t,\t%s" % (out, err)
            command = "convert -resize 550x550 \"%s\" \"%s\"" % tuple([filename] * 2)
            print "Montage resize command: %s" % command
            out, err = Popen(split(command)).communicate()
            print "Montage resize output: %s\t,\t%s" % (out, err)

            # filename = os.path.join(os.getcwd(), Constants.MEANING_DIR,
            # "%d%s_resized.%s" % (self.phase + 1, dimension, Constants.IMG_EXTENSION))
            # print "Image: %s" % filename
            pixmap = QtGui.QPixmap(filename)
            # print "Pixmap: %s" % pixmap
            label.setPixmap(pixmap)
            remove(filename)
        label.repaint()

        # connect the button clicked signal
        button = self.infoWindow.findChildren(QtGui.QPushButton, "btnOkay")[0]
        disconnect(button)

        # self.connection.factory.mode == Constants.LEARN:
        if mode == constants.MOD_PREPHASE:
            assert self.connection.factory.mode == constants.LEARN
            connect(
                button, "clicked()", lambda: self.send(constants.REQ_NEXT_PIC))
            # connect(button, "clicked()", lambda : self.show_window(self.learningWindow))
        # self.connection.factory.mode == Constants.TEST:
        elif mode == constants.MOD_PRETEST:
            assert self.connection.factory.mode == constants.TEST
            connect(
                button, "clicked()", lambda: self.show_window(self.testWindow))
        elif mode == constants.MOD_EXIT:
            connect(
                button, "clicked()", lambda: QtCore.QCoreApplication.exit())
        elif mode == constants.MOD_FIRSTSCREEN:
            connect(
                button, "clicked()", lambda: self.show_window(self.infoWindow,
                                                              mode=constants.MOD_PREPHASE))

    def setupTestWindow(self):
        get = getFunction(self.testWindow)
        btn = get(QtGui.QPushButton, "btnSubmit")
        group = get(QtGui.QGroupBox, "groupBox")
        get(QtGui.QPushButton, "btnPlay").setEnabled(True)

        def submit():
            get(QtGui.QPushButton, "btnPlay").setEnabled(False)
            given_answer = self.current_question.given_answer
            if given_answer is None:
                print "Cannot submit before participant chooses an answer."
                get(QtGui.QPushButton, "btnPlay").setEnabled(True)
                return
            success = (self.target_meaning == self.given_meaning)
            assert given_answer is not None
            assert success == (
                self.current_question.given_answer == self.current_question.answer)
            print "Given:", given_answer, "Expected:", self.images[self.phase].index(self.given_meaning)
            assert given_answer == self.images[
                self.phase].index(self.given_meaning)
            self.send(constants.START_RESPONSE)
            print "Sending the answer %d" % self.current_question.given_answer
            self.send(encode(self.current_question.given_answer))
            self.send(constants.END_RESPONSE)
            # get(QtGui.QGroupBox, "groupBox").setEnabled(False)
            if not success:
                btn = self.getCheckedButton()
                setButtonIcon(
                    btn, btn.meaning.pixmap((255, 0, 0, 140)))
                # disconnect(btn)

            setButtonIcon(
                self.correct_button, self.correct_button.meaning.pixmap((0, 255, 0, 180)))
            self.correct_button = None
            self.given_meaning = None
            self.target_meaning = None

            def next():
                get(QtGui.QPushButton, "btnPlay").setEnabled(True)
                self.next_question()

            # self.next_question)
            QtCore.QTimer.singleShot(constants.DELAY_TEST, next)

        disconnect(btn)
        connect(btn, "clicked()", submit)

        btn.setEnabled(False)

        for i in range(1, 5):
            btn_ = get(QtGui.QPushButton, "pushButton_%d" % (i))
            btn_.setAutoExclusive(False)
            btn_.setChecked(False)
            btn_.setAutoExclusive(True)
        self.theremin.mute()

        play = get(QtGui.QPushButton, 'btnPlay')

        def enable_submit():
            btn.setEnabled(True)

        connect(play, "clicked()", enable_submit)

    def on_new_picture(self, data):
        """
        Callback for when a new picture is received from
        the server.
        """
        pic = decode(data)
        # print "New picture: %s" % pic
        self.show_window(self.learningWindow, meaning=pic)

    def mute(self):
        self.theremin.player.mute()
        self.playback_player.player.mute()
        self.muted = True

    def unmute(self):
        self.theremin.player.unmute()
        self.playback_player.player.unmute()
        self.muted = False

    def show_window(self, window, full_screen=True, meaning=None, mode=None):
        self.setupWindow(window, meaning, mode)

        if self.activeWindow and self.activeWindow != window:
            self.activeWindow.hide()
        self.activeWindow = window

        if full_screen:
            window.showFullScreen()
        else:
            window.show()

            # print self.connection.factory.mode

    def start_recording(self):
        self.last_signal = []
        self.isRecording = True
        self.theremin.record()

    def stop_recording(self):
        self.isRecording = False
        self.theremin.stop_record()
        self.send(constants.START_REC)
        for frame in self.last_signal:
            self.send(frame)
        self.send(constants.END_REC)

    def next_phase(self, practice=False):
        self.isPractice = practice
        if self.isPractice:
            self.phase += 1
        print "Phase %d" % self.phase
        if self.phase == 0 and practice:
            self.show_window(self.infoWindow, mode=constants.MOD_FIRSTSCREEN)
        else:
            self.show_window(self.infoWindow, mode=constants.MOD_PREPHASE)

    def extend_last_signal(self, pickled_frame):
        if self.isRecording:
            self.last_signal.append(pickled_frame)

    def next_question(self):
        self.last_signal = []
        self.send(constants.REQ_NEXT_PIC)

    def go_test(self):
        self.show_window(self.infoWindow, mode=constants.MOD_PRETEST)
        # self.show_window(self.testWindow)

    def getCheckedButton(self):
        get = getFunction(self.testWindow)
        for i in range(1, 5):
            btn = get(QtGui.QPushButton, "pushButton_%d" % (i))
            if btn.isChecked():
                return btn

    def on_new_test_question(self, question):
        # this makes sure we don't lose the question during
        # the pre-test info screen
        if self.activeWindow != self.testWindow:
            # print "Waiting for test window"
            QtCore.QTimer.singleShot(.5, lambda: self.on_new_test_question(
                question))
            return
        self.setupWindow(self.testWindow)
        self.current_question = question
        self.correct_button = None
        get = getFunction(self.testWindow)
        get(QtGui.QGroupBox, "groupBox").setEnabled(True)
        print "Pics: %s" % question.pics
        print "Answer: %s" % question.answer

        disconnect(self.buttonSignalMapper)
        self.buttonSignalMapper = QtCore.QSignalMapper(self.testWindow)

        @QtCore.pyqtSlot(int)
        def on_select(meaning_id):
            print "Selected %d" % meaning_id
            self.current_question.given_answer = meaning_id
            self.given_meaning = self.images[self.phase][meaning_id]
            # get(QtGui.QPushButton, "btnSubmit").setEnabled(True)

        for i, meaning in zip(range(1, len(question.pics) + 1), question.pics):
            print i, meaning

            btn = get(QtGui.QPushButton, "pushButton_%d" % (i))
            meaning_obj = self.images[self.phase][meaning]
            setButtonIcon(
                btn, self.images[self.phase][meaning].pixmap())
            btn.meaning = self.images[self.phase][meaning]
            # btn.setText(str(meaning_obj) + "(%d)" % meaning)

            disconnect(btn)
            connect(btn, signal="clicked()", slot="map()",
                    widget2=self.buttonSignalMapper)

            self.buttonSignalMapper.setMapping(btn, int(meaning))
            if meaning == question.answer:
                self.correct_button = btn
                self.target_meaning = self.images[self.phase][meaning]
                # btn.setStyleSheet("background-color: rgb(255, 255, 255);")
        disconnect(self.buttonSignalMapper)
        connect(self.buttonSignalMapper, signal="mapped(int)", slot=on_select)
        self.last_signal = question.signal
        print question

    def flicker(self):
        """
        This method simply jiggles the mouse. It is used as
        part of a workaround for LoopingCall's getting stuck
        on the first call when using qt4reactor on linux.
        """
        from platform import system

        if (self.activeWindow is not None) and system() == "Linux":
            from pymouse import PyMouse

            m = PyMouse()
            x, y = m.position()
            m.move(x + 1, y)

    def send(self, msg):
        """
        Sends the message to the server. Convenience method.
        """
        self.theremin.protocol.sendLine(msg)
        if "LeapFrame" not in msg:
            # msg = "LeapFrame()"
            print "Sent: %s" % msg

    def exit(self):
        self.show_window(window=self.infoWindow, mode=constants.MOD_EXIT)


if __name__ == "__main__":
    import sys

    ui = ClientUI('1')
    print "Running the reactor"
    ui.reactor.runReturn()
    print "Launching QApplication..."
    code = app.exec_()
    print "Done"
    # print ui.last_signal
    sys.exit(code)
