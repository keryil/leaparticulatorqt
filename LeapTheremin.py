#!/usr/bin/python
import sys
from collections import deque

from twisted.python import log
import jsonpickle

import Leap
from Tone import Tone
from leaparticulator import constants
from LeapFrame import LeapFrame

from PySide import QtCore, QtGui #, QtUiTools
from leaparticulator.constants import install_reactor
install_reactor()

# if "qt4reactor" not in sys.modules:
#     import qt4reactor
#     from PySide.QtGui import QApplication
#     qapplication = QApplication(sys.argv)
#     qt4reactor.install()
# from twisted.internet import reactor

# this resolves bulk sending of frames when using mac
# if system() == "Darwin":
#   from twisted.internet import cfreactor
#   cfreactor.install()
# else:
#   from twisted.internet import gtk2reactor
#   gtk2reactor.install()
# if "twisted.internet.reactor" not in sys.modules:
#     from twisted.internet import gtk2reactor
#     gtk2reactor.install()
from twisted.internet import reactor
from twisted.internet.task import LoopingCall

from LeapServer import LeapClientFactory
from leaparticulator.constants import palmToAmpAndFreq
# import pygtk
# pygtk.require("2.0")

# import gtk#, gtk.glade

signal = []


class ThereminPlayer(object):

    """
    This is the main class used to play the theremin. 
    It takes LeapFrame objects and produces sound using 
    a PyAudio synth.
    """
    default_volume = 0
    volume_coefficient = 1.
    muted = False
    fadeout_call = None
    fadeout_counter = 0
    default_pitch = None
    ui = None

    def __init__(self, n_of_tones=1, default_volume=.5,
                 default_pitch=None, ui=None):
        tones = []
        self.ui = ui
        for i in range(n_of_tones):
            t = Tone()
            t.open()
            t.setAmplitude(0)
            t.start()
            tones.append(t)
        self.tones = tones
        self.default_pitch = default_pitch
        self.default_volume = default_volume
        log.startLogging(sys.stdout)

    def fadeOut(self):
        # import traceback
        # traceback.print_stack()
        self.fadeout_counter += 1
        for tone in self.tones:
            amp = tone.getAmplitude()
            # print("Fadeout call %i: new amp is %f, with delta %f" % (self.fadeout_counter,
            #                                                       amp *Constants.fadeout_multiplier,
            #                                                       Constants.fadeout_multiplier))
            # print("Next call at %s" % self.fadeout_call._expectNextCallAt)
            if amp > 0.005:
                amp *= constants.fadeout_multiplier
            else:
                amp = 0
                self.resetFadeout()
            tone.setAmplitude(amp)

        # this bit is a workaround for LoopingCall
        # getting stuck on the first call when using
        # qt4reactor
        if self.ui and self.fadeout_counter == 1:
            self.ui.flicker()

    def resetFadeout(self):
        if self.fadeout_call is not None:
            if self.fadeout_call.running:
                self.fadeout_call.stop()
            self.fadeout_counter = 0
        # self.fadeout_call = None

    def newPosition(self, frame):
        """
        Calculates the new amp and frequency values given the frame,
        changes the playing tone accordingly, and returns the corresponding
        values.
        """
        amp, freq = 0, 0

        def on_error(err):
            print err

        if not frame.hands or self.muted:
            if self.fadeout_call is None:
                # print "HELLOOOO"
                # ,clock=reactor)
                self.fadeout_call = LoopingCall(f=self.fadeOut)
                # self.fadeout_call.addErrback(twisted.python.log.err)
                self.fadeout_call_def = self.fadeout_call.start(
                    constants.fadeout_call_rate)
                self.fadeout_call_def.addErrback(on_error)
                # print self.fadeout_call
            # else:
            #   [tone.setAmplitude(0) for tone in self.tones]
        elif len(self.tones) == 1:
            self.resetFadeout()
            self.fadeout_call = None

            hand = frame.hands[0].stabilized_palm_position
            amp, freq = palmToAmpAndFreq(hand)
            if amp == freq == 0:
                return 0, 0
            if self.default_volume:
                amp = self.default_volume
            else:
                # use the max() trick to make sure we don't have
                # zero volume for 2d cases
                amp = max(0.0001, amp)
            if self.default_pitch:
                freq = self.default_pitch
            # print amp, freq
            self.tones[0].setAmplitude(0)
            self.tones[0].setFrequency(freq)
            self.tones[0].setAmplitude(amp * self.volume_coefficient)

#           print frame.timestamp
            # log.msg("Playing: freq=%s, amp=%s, mel=%s, timestamp=%s" % (freq,
            #                                                   amp,
            #                                                   freqToMel(freq),
            #                                                   frame.timestamp))
        return amp, freq

    def mute(self):
        self.muted = True
        for t in self.tones:
            t.setAmplitude(0)

    def unmute(self):
        self.muted = False

    def setVolume(self, value):
        assert 0 <= value <= 1
        self.volume_coefficient = value


class ThereminPlayback(object):

    """
    This class is used to replay sounds produced by LeapTheremin from
    recorded frames of the Leap controller. The recorded frames are given
    to the constructor as the parameter "score", and is expected to be an
    iterable of LeapFrame objects. It uses the average frame rate (per second)
    for playback.
    """

    player = None
    rate = None
    score = None
    counter = 0
    call = None

    def __init__(self, n_of_tones=1, default_volume=.5):
        self.player = ThereminPlayer(n_of_tones, default_volume)
        self.player.mute()

    def setVolume(self, value):
        self.player.setVolume(value)

    def play(self):
        # log.msg("Frame %s of this recording." % self.counter)
        try:
            f = self.score[self.counter]
            self.player.newPosition(f)
            self.counter += 1
        except IndexError, e:
            # print e
            # print "Ran out of frames, stopping playback..."
            self.stop()
        # finally:

    def start(self, score, callback=None):
        if self.call:
            if self.call.running:
                self.stop()
            else:
                self.call = None
        self.callback = callback
        # self.call = QtCore.QTimer()

        self.call = LoopingCall(self.play)  # , self)
        # print "Score is something like: ", score
        self.score = deque([jsonpickle.decode(f) for f in score])
        # for f in self.score:
        #   print f.current_frames_per_second
        fps = (f.current_frames_per_second for f in self.score)
        average_fps = float(sum(fps)) / len(self.score)
        self.rate = 1. / average_fps
        # self.rate = 1. / min([f.current_frames_per_second for f in self.score])
        self.player.unmute()
        self.call.start(self.rate, now=True).addErrback(log.err)

    def stop(self):
        # try:
        if self.call:
            if self.call.running:
                self.call.stop()
            self.call = None
            if self.callback:
                self.callback()
            # except AssertionError, err:
            #   print err
            # raise err
            # finally:
            self.player.mute()
            self.counter = 0
            # self.call = None
            print "Stopped"


def gimmeSomeTheremin(n_of_notes, default_volume, ip=constants.leap_server,
                      ui=None, factory=LeapClientFactory, realtime=True,
                      uid=None):
    """
    Returns an initialized theremin, its reactor object, and the connection
    (protocol) object.
    """
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_BACKGROUND_FRAMES)
    print "Init the theremin listener"
    audio_listener = ThereminListener(n_of_tones=1, default_volume=default_volume,
                                      realtime=realtime, ui=ui)
    print "Done"
    connection = None
    if ip != None:
        log.msg("Connecting to server at %s:%s" % (ip, constants.leap_port))
        connection = None
        if uid is None:
            connection = reactor.connectTCP(
                ip, constants.leap_port, factory(audio_listener, ui))
        else:
            connection = reactor.connectTCP(
                ip, constants.leap_port, factory(audio_listener, ui, uid))

    # point = TCP4ClientEndpoint(reactor, ip, Constants.leap_port)
    # connection = point.connect(LeapClientFactory(audio_listener, ui))
    # log.msg("Connected")
        # Have the audio_listener receive events from the controller
    controller.add_listener(audio_listener)
    return audio_listener, reactor, controller, connection


class ThereminListener(Leap.Listener):

    """
    This class is used to listen to Leap Motion input and 
    use an ThereminPlayer instance to play the corresponding 
    tones. This is glue between the Leap and the theremin, much 
    like the ThereminPlayback class is glue between recorded Leap
    frames and the theremin. 
    """
    last_timestamp = -1
    default_volume = 0
    protocol = None
    player = None
    factory = None
    recording = False
    last_signal = []
    callback = None

    def __init__(self, n_of_tones=1, default_volume=.5, realtime=True, ui=None):
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui)
#       self.tones = tones
#       self.default_volume = default_volume
        log.startLogging(sys.stdout)
#       self.protocol = protocol

    def on_frame(self, controller):
        """
        This is the callback for Leap Motion
        """
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        pickled = jsonpickle.encode(LeapFrame(frame))
        # print "Frame:", frame
        timestamp = 0
        if self.last_timestamp != -1:
            timestamp = frame.timestamp - self.last_timestamp

        self.last_timestamp = frame.timestamp
        # print "New frame"
        amp, freq = self.player.newPosition(frame)
        if self.callback:
            # print "Calling"
            self.callback((amp, freq))

        if (not self.player.muted) \
                and (amp != 0)\
                and (freq != 0):
            if self.protocol:
                if self.realtime:
                    self.protocol.sendLine(pickled)
                # else:
                #     log.msg("Extending signal...")
                if self.protocol.factory.ui:
                    self.protocol.factory.ui.extend_last_signal(pickled)
                    # self.protocol.factory.extendSignal(pickled)
            else:
                if not self.realtime:
                    if self.recording:

                        # log.msg("Extending signal...")
                        self.last_signal.append(pickled)
                    # self.factory.ui.extendSignal(pickled)

    def record(self):
        self.recording = True

    def stop_record(self):
        self.recording = False

    def get_signal(self):
        return self.last_signal

    def reset_signal(self):
        self.last_signal = []

    def mute(self):
        self.player.mute()

    def unmute(self):
        self.player.unmute()

    def setVolume(self, value):
        self.player.setVolume(value)


def main(ip=constants.leap_server):
    tones = []
    theremin = None
    connection = None
    controller = None

    def stop():
        reactor.stop()
        controller.remove_listener(theremin)
    try:
        theremin, reactor, controller, connection = gimmeSomeTheremin(
            n_of_notes=1, default_volume=.5, ip=ip)
        reactor.callLater(100, stop)
        if not constants.TEST:
            print "Starting reactor"
            reactor.run()

    finally:
        for f in signal:
            print f

if __name__ == "__main__":
    #   lines = open("frames.list").read().split("\n")[:-1]
    # print "Last line is:", lines[-1]
    # sys.exit()
    # score = []
    #   for line in lines:
    #       print "Line:", line
    # score.append(jsonpickle.decode(line))
    # score = [jsonpickle.decode(f) for f in lines]
    #   p = ThereminPlayback()
    #   p.start(lines)
    #   print "starting reactor..."
    #   reactor.run()

    #   main(sys.argv[1])
    # gimmeSomeTheremin(n_of_notes=1, default_volume=.5,ip=None)
    # print "Done!"
    # reactor.run()
    # main(ip=None)
    # import sys
    # install_reactor()

    theremin, reactor, controller, connection = gimmeSomeTheremin(n_of_notes=1, default_volume=.5,
                                                                  ip=None)
    if not constants.TEST:
        reactor.runReturn()
    app = QtGui.QApplication.instance()
    sys.exit(app.exec_())
    # class Frame(object):
    #   def __init__(self, x, y):
    #       class Pos(object):
    #           stabilized_palm_position = None
    #       p = Pos()
    #       p.stabilized_palm_position = (x,y,0)
    #       self.hands = [p]

    # app = QtGui.QApplication.instance()
    # if app is None:
    #     app = QtGui.QApplication(sys.argv)
    # else:
    #     print "ExistingQ Application instance:", app

    # controller = Leap.Controller()
    # controller.set_policy_flags(Leap.Controller.POLICY_BACKGROUND_FRAMES)

    # player = ThereminListener()
    # controller.add_listener(player)
    # def fn(arg):
    #   print arg
    # player.callback = fn

    # for i, j in zip(range(0,300,100), [100] * 3):
    # player.newPosition(Frame(i,j))
    # from time import sleep
    # sleep(1)
    # r

    # print "Running the reactor"
    # MainWindow = ui.learningWindow
    # MainWindow.show()
    # MainWindow.reactor.run()
    # print "Done"
