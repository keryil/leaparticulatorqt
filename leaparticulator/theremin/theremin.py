import sys

from twisted.python import log
import jsonpickle

import Leap
from leaparticulator.data.frame import LeapFrame
from leaparticulator.constants import install_reactor, palmToAmpAndFreq, freqToMel
import leaparticulator.constants as constants
from leaparticulator.theremin.tone import Tone

install_reactor()
from twisted.internet.task import LoopingCall
from twisted.internet import reactor


class ThereminPlayer(object):
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
            print("Fadeout call %i: new amp is %f, with delta %f" % (self.fadeout_counter,
                amp, constants.fadeout_multiplier))

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

            log.msg("Playing: freq=%s, amp=%s, mel=%s, timestamp=%s" % (freq,
                                                                        amp,
                                                                        freqToMel(freq),
                                                                        frame.timestamp))
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

    def __init__(self, n_of_tones=1, default_volume=.5, realtime=True, ui=None):
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui)
        self.controller = Leap.Controller()
        self.controller.add_listener(self)
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
    def __init__(self, n_of_tones=1, default_volume=.5, realtime=True, ui=None):
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui)
        self.controller = Leap.Controller()
        log.startLogging(sys.stdout)
        self.call = LoopingCall(self.on_frame, self.controller)
        self.call.start(interval=constants.theremin_rate)

    def stop(self):
        self.call.stop()



if __name__ == "__main__":
    # theremin = Theremin()
    theremin = ConstantRateTheremin()
    reactor.run()