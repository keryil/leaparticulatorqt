__author__ = 'Kerem'
import math
import array
import threading
import contextlib
import wave
from ctypes import cdll, c_char_p, c_int, CFUNCTYPE

import pyaudio

from leaparticulator import constants


ERROR_HANDLER_FUNC = CFUNCTYPE(
    None,
    c_char_p,
    c_int,
    c_char_p,
    c_int,
    c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass


c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextlib.contextmanager
def noalsaerr():
    """
    A context manager that suppresses ALSA Warnings
    """
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except OSError:
        yield


class Tone(object):
    def __init__(self, rate=constants.AUDIO_FRAMERATE, frequency=constants.default_pitch,
                 amplitude=constants.default_amplitude, record=False, filename=None):
        self.record = record
        if record:
            self.recorded_frames = []
        self.rate = rate
        self.freq = frequency
        self.amp = amplitude
        self.phase = 0.0

    def open(self):
        with noalsaerr():
            self.p = pyaudio.PyAudio()
        self.stream = self.p.open(rate=int(self.rate), channels=1,
                                  format=constants.THEREMIN_AUDIO_FORMAT, output=True,
                                  frames_per_buffer=constants.FRAMES_PER_BUFFER)
        if self.record:
            print self.p.get_default_output_device_info()
            # SPEAKERS = self.p.get_default_output_device_info()["hostApi"] #The part I have modified

            # self.readfrom_stream = self.p.open(format=constants.THEREMIN_AUDIO_FORMAT,
            #                 channels=2,
            #                 rate=int(self.rate),
            #                 input=True,
            #                 frames_per_buffer=constants.FRAMES_PER_BUFFER)
            #                 # input_host_api_specific_stream_info=SPEAKERS) #The part I have modified
    def close(self):
        self.stream.close()
        # self.readfrom_stream.close()
        self.p.terminate()

    def setAmplitude(self, amp):
        self.amp = amp

    def getAmplitude(self):
        return self.amp

    def setFrequency(self, freq):
        self.freq = freq

    def start(self):
        self.t = threading.Thread(target=self.run)
        self.t.daemon = True
        self.running = True
        self.t.start()

    def stop(self):
        self.running = False
        self.t.join()

    def run(self):
        def gen():
            for i in xrange(int(self.rate * 0.05)):
                if constants.THEREMIN_AUDIO_FORMAT == pyaudio.paInt32:
                    yield int((math.sin(self.phase) + 1) * self.amp * (2 ** 31))
                else:
                    yield math.sin(self.phase) * self.amp
                self.phase += 2. * math.pi * self.freq / self.rate
                if self.phase > math.pi:
                    self.phase -= 2. * math.pi

        if not constants.TESTING:
            while self.running:
                type = 'f'
                if constants.THEREMIN_AUDIO_FORMAT == pyaudio.paInt32:
                    type = 'I'
                buf = array.array(type, gen())
                self.stream.write(buf.tostring())

                # free = self.stream.get_write_available() # How much space is left in the buffer?

                if self.record:
                    # in_buf = self.readfrom_stream.read(constants.FRAMES_PER_BUFFER)
                    self.recorded_frames.append(buf.tostring())

                    # SILENCE = 0
                    # if free > constants.FRAMES_PER_BUFFER: # Is there a lot of space in the buffer?
                    #     print "Filling in"
                    #     tofill = free - constants.FRAMES_PER_BUFFER
                    #     self.stream.write(SILENCE * tofill)
                    #     self.recorded_frames.append(SILENCE * tofill)
                    # print buf

    def dump_to_file(self, filename):
        if self.record:
            # import sounddevice as sd
            # import numpy as np
            # frames = sd.playrec(np.asarray(self.recorded_frames, dtype=np.int32), constants.AUDIO_FRAMERATE, blocking=True)
            wf = wave.open(filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(constants.THEREMIN_AUDIO_FORMAT))
            wf.setframerate(int(self.rate))
            # print self.recorded_frames[0]
            # print self.recorded_frames[10]
            wf.writeframes(b''.join(self.recorded_frames))
            wf.close()
            self.reset_record()
        else:
            print "Cannot dump Tone to file, this is not a recording playback instance."

    def reset_record(self):
        if self.record:
            self.recorded_frames = []


if __name__ == '__main__':
    import time

    t = Tone(frequency=462)
    t.open()
    t.start()
    time.sleep(2)
    t.stop()
    t.close()
