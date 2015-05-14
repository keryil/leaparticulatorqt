import sys

from twisted.python import log
import jsonpickle

import Leap
from leaparticulator.data.frame import LeapFrame
from leaparticulator.constants import install_reactor, palmToAmpAndFreq
import leaparticulator.constants as constants
from leaparticulator.theremin.tone import Tone
from LeapServer import LeapClientFactory
from collections import deque

install_reactor()
from twisted.internet.task import LoopingCall
from twisted.internet import reactor


class ThereminPlayer(object):
    """
    This is what produces the sounds of the theremin.
    """
    def __init__(self, n_of_tones, default_volume=constants.default_amplitude, default_pitch=None,
                 ui=None):
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

        self.volume_coefficient = 1.
        self.muted = False
        self.fadeout_call = None
        self.fadeout_counter = 0
        log.startLogging(sys.stdout)

    def fadeOut(self):
        self.fadeout_counter += 1
        for tone in self.tones:
            amp = tone.getAmplitude()
            # print("Fadeout call %i: new amp is %f, with delta %f" % (self.fadeout_counter,
            #     amp, constants.fadeout_multiplier))

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

    def newPosition(self, frame):
        """
        Calculates the new amp and frequency values given the frame,
        changes the playing tone accordingly, and returns the corresponding
        values.
        """
        amp, freq = 0, 0

        def on_error(err):
            log.err(err)

        if not frame.hands or self.muted:
            if self.fadeout_call is None:
                self.fadeout_call = LoopingCall(f=self.fadeOut)
                # self.fadeout_call.addErrback(twisted.python.log.err)
                self.fadeout_call_def = self.fadeout_call.start(
                    constants.fadeout_call_rate)
                self.fadeout_call_def.addErrback(on_error)

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

            self.tones[0].setAmplitude(0)
            self.tones[0].setFrequency(freq)
            self.tones[0].setAmplitude(amp * self.volume_coefficient)

            # log.msg("Playing: freq=%s, amp=%s, mel=%s, timestamp=%s" % (freq,
            #                                                             amp,
            #                                                             freqToMel(freq),
            #                                                             frame.timestamp))
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


class Theremin(Leap.Listener):
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

    def __init__(self, n_of_tones=1, default_volume=.5, ui=None,
                 factory=LeapClientFactory, realtime=False):
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui)
        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_BACKGROUND_FRAMES)
        self.controller.add_listener(self)
        print "Connecting to: %s:%s" % (constants.leap_server,
                                        constants.leap_port)
        self.protocol = reactor.connectTCP(constants.leap_server,
                                           constants.leap_port,
                                           factory(self, ui))
        # self.reactor = reactor
        log.startLogging(sys.stdout)

    def on_init(self, controller):
        log.msg("LeapController initialized")

    def on_connect(self, controller):
        log.msg("Leap connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        log.msg("Leap disconnected")

    def on_exit(self, controller):
        log.msg("LeapController exited")

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
                and (amp != 0) \
                and (freq != 0):
            if self.protocol:
                if self.realtime:
                    self.protocol.sendLine(pickled)
                if self.protocol.factory.ui:
                    self.protocol.factory.ui.extend_last_signal(pickled)
            else:
                if not self.realtime and self.recording:
                        self.last_signal.append(pickled)

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

class ConstantRateTheremin(Theremin):
    def __init__(self, n_of_tones=1, default_volume=.5, realtime=True, ui=None,
                 rate=constants.theremin_rate):
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.rate = rate
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui)
        self.controller = Leap.Controller()
        # self.reactor = reactor
        log.startLogging(sys.stdout)
        self.call = LoopingCall(self.on_frame, self.controller)

    def start(self):
        if not self.call.running:
            self.call.start(interval=self.rate)

    def stop(self):
        self.call.stop()

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

    def __init__(self, n_of_tones=1, default_volume=.5, default_rate=None):
        self.player = ThereminPlayer(n_of_tones, default_volume)
        self.player.mute()
        self.default_rate = default_rate

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
        if self.default_rate:
            self.rate = self.default_rate
        else:
            fps = (f.current_frames_per_second for f in self.score)
            average_fps = float(sum(fps)) / len(self.score)
            self.rate = 1. / average_fps

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

if __name__ == "__main__":
    # theremin = Theremin()
    print "theremin.py test code init..."
    theremin = ConstantRateTheremin()
    reactor.run()