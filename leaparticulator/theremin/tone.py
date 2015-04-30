import math
import array
import threading
import contextlib
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
                 amplitude=constants.default_amplitude):
        self.rate = rate
        self.freq = frequency
        self.amp = amplitude
        self.phase = 0.0

    def open(self):
        with noalsaerr():
            self.p = pyaudio.PyAudio()
        self.stream = self.p.open(rate=int(self.rate), channels=1,
                                  format=pyaudio.paFloat32, output=True,
                                  frames_per_buffer=constants.FRAMES_PER_BUFFER)


    def close(self):
        self.stream.close()
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
                yield math.sin(self.phase) * self.amp
                self.phase += 2. * math.pi * self.freq / self.rate
                if self.phase > math.pi:
                    self.phase -= 2. * math.pi

        if not constants.TEST:
            while self.running:
                buf = array.array('f', gen()).tostring()
                self.stream.write(buf)


if __name__ == '__main__':
    import time

    t = Tone(frequency=462)
    t.open()
    t.start()
    time.sleep(2)
    t.stop()
    t.close()
