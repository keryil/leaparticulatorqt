import platform
import sys

import jsonpickle
from twisted.python import log


"""
This piece of code imports the Leap libraries. It is necessary to change the hard
path in the library file under OS/X, hence all this mambo-jamboa
"""
if 'Leap' not in sys.modules:
    if platform.system() == "Linux":
        import leaparticulator.drivers.linux.Leap as Leap
    else:
        # let's make the otools stuff automatic
        # find the .so file
        from os.path import dirname, join, abspath
        from os import walk, sep
        import subprocess
        import fnmatch

        dir = sep + join(*(abspath(dirname(__file__)).split(sep) + ["..", "drivers", "osx"]))
        f = join(dir, "LeapPython.so")

        # get current info
        command = "otool -L %s" % f
        output = subprocess.check_output(command.split())
        first_path = output.split()[1].lstrip()
        done = False
        # find the libpython2.7.dylib
        for root, dnames, fnames in walk('/usr/local/Cellar/python'):
            for fname in fnmatch.filter(fnames, "libpython2.7.dylib"):
                # act on the first file
                # update the info
                dylib_path = join(root, fname)
                command = "install_name_tool -change %s %s %s" % (first_path,
                                                                  dylib_path,
                                                                  f)
                print("Issuing command: %s" % command)
                subprocess.check_call(command.split())
                done = True
                break
            if done:
                break

        import leaparticulator.drivers.osx.Leap as Leap

from leaparticulator.data.frame import LeapFrame
from leaparticulator.constants import install_reactor, palmToAmpAndFreq

# ensure reactor is installed at this point
install_reactor()

import leaparticulator.constants as constants
from leaparticulator.theremin.tone import Tone
from leaparticulator.oldstuff.LeapServer import LeapClientFactory
from collections import deque

from twisted.internet.task import LoopingCall
from twisted.internet import reactor

class ThereminPlayer(object):
    """
    This is what produces the sounds of the theremin. It can have variable pitch and amplitude,
    as well as produce polytonic signals.
    """
    def __init__(self, n_of_tones, default_volume=constants.default_amplitude, default_pitch=None,
                 ui=None, record=False):
        tones = []
        self.ui = ui
        for i in range(n_of_tones):
            t = Tone(record=record)
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

    def dumpRecording(self, files):
        """
        Dump recording to a wave file.
        :param files:
        :return:
        """
        for tone, f in zip(self.tones, files):
            tone.dump_to_file(f)

    def fadeOut(self):
        """
        Fadeout the volume using a separate job. Helps reduce pops.
        :return:
        """
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
        """
        Resets the counter for the fadeout job.
        :return:
        """
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
        """

        :param n_of_tones: Number of simultaneous tones the theremin will produce.
        :param default_volume: Fixed volume, set to None for variable volume.
        :param ui: The ClientUI object that will control the theremin.
        :param factory: The twisted reactor that runs the theremin.
        :param realtime: If set to true, the theremin will broadcast the frames to the server in real time.
        """
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui)
        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_BACKGROUND_FRAMES)
        self.controller.add_listener(self)
        print("Connecting to: %s:%s" % (constants.leap_server,
                                        constants.leap_port))
        if factory:
            self.protocol = reactor.connectTCP(constants.leap_server,
                                           constants.leap_port,
                                           factory(self, ui))
        else:
            self.protocol = None
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
                    # print "extending"
                    self.protocol.factory.ui.extend_last_signal(pickled)
            else:
                if not self.realtime and self.recording:
                    # print "appending"
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
    """
    This is a theremin that polls the Leap and produces frames at a constant rate.
    """
    def __init__(self, n_of_tones=1, default_volume=.5, realtime=True, ui=None,
                 rate=constants.THEREMIN_RATE, factory=LeapClientFactory):
        Leap.Listener.__init__(self)
        self.realtime = realtime
        self.rate = rate
        self.player = ThereminPlayer(n_of_tones=n_of_tones,
                                     default_volume=default_volume,
                                     ui=ui,
                                     default_rate=rate)
        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_BACKGROUND_FRAMES)

        print("Connecting to: %s:%s" % (constants.leap_server,
                                        constants.leap_port))
        self.protocol = reactor.connectTCP(constants.leap_server,
                                           constants.leap_port,
                                           factory(self, ui))
        # self.reactor = reactor
        log.startLogging(sys.stdout)
        self.call = LoopingCall(self.on_frame, self.controller)
        self.start()

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
    stopping = False
    late_callbacks = []
    callback = None

    def __init__(self, n_of_tones=1, default_volume=.5, default_rate=None, record=False):
        self.player = ThereminPlayer(n_of_tones, default_volume, record=record)
        self.player.mute()
        self.default_rate = default_rate
        self.record = record
        self.filename = None

    def setVolume(self, value):
        self.player.setVolume(value)

    def play(self):
        try:
            frame = self.score[self.counter]
            self.player.newPosition(frame)
            self.counter += 1
        except IndexError:
            self.stop()

    def start(self, score, callback=None, filename=None, jsonencoded=True):
        """
        Start the playback
        :param score: The list of possibly json encoded frames to play.
        :param callback: The function to call when the playback is done.
        :param filename: The filename to which the generated audio will be dumped, if any.
        :param jsonencoded: Whether the score is a list of json encoded frames or frame objects.
        :return:
        """
        from copy import deepcopy as copy
        if self.record:
            assert filename
            self.filename = filename
        if self.call:
            if self.call.running:
                self.call.stop()
            else:
                self.call = None
        self.callback = callback
        # self.call = QtCore.QTimer()

        self.counter = 0
        self.call = LoopingCall(self.play)  # , self)
        # print "Score is something like: ", score
        if jsonencoded:
            self.score = deque([jsonpickle.decode(f) for f in copy(score)])
        else:
            self.score = deque(copy(score))
        # for f in self.score:
        #   print f.current_frames_per_second
        if self.default_rate:
            self.rate = self.default_rate
        else:
            fps = (f.current_frames_per_second for f in self.score)
            average_fps = float(sum(fps)) / len(self.score)
            self.rate = 1. / average_fps

        self.player.unmute()
        from datetime import datetime
        self.start_time = datetime.now()
        self.call.start(self.rate, now=True).addErrback(log.err)

    def stop(self):
        # try:
        if self.call and not self.stopping:
            # submit a fake frame to trigger the fadeout
            fake_frame = LeapFrame(None, random=True)
            fake_frame.hands = []
            self.player.newPosition(fake_frame)
            if self.call.running:
                self.call.stop()

            self.counter = 0

            def finalize():
                self.call = None
                # except AssertionError, err:
                #   print err
                # raise err
                # finally:
                self.player.mute()
                self.player.dumpRecording([self.filename])
                # self.call = None
                self.stopping = False
                print("Stopped")

                from datetime import datetime
                print(datetime.now() - self.start_time)
                if self.callback:
                    reactor.callLater(0, self.callback)
                    for callback in self.late_callbacks:
                        reactor.callLater(0, callback)
                    # self.callback()
            self.stopping = True
            # allow a short time for the fadeout to end
            reactor.callLater(.5, finalize)


if __name__ == "__main__":
    # theremin = Theremin()
    print("theremin.py test code init...")
    theremin = ConstantRateTheremin()
    reactor.run()