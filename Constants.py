'''
Created on Feb 21, 2014

@author: kerem
'''

from os.path import join

# units
default_pitch=440.
default_amplitude=.5

leap_port = 9999
leap_server = "127.0.0.1"
# leap_server = "134.184.26.67"
# leap_server = "134.184.26.61"
# leap_server = "127.0.0.1"

fadeout_multiplier = 0.95
fadeout_call_rate = 0.005

# number of options each test question has
n_of_options = [4, 4, 4]
# number of total available images per phase
n_of_meanings = [5,10,15]
# number of test questions per phase
n_of_test_questions = n_of_meanings

question_mark_path = join("img","question_mark.jpg")

# signals
SUBMIT = "Submit"

REQ_NEXT_PIC = "Request picture"

START_NEXT_PIC = "Start picture"
END_NEXT_PIC = "End picture"

IMAGE_LIST = "Image list"

START = "Start"
STOP = "Stop"

START_REC = "Start recording"
END_REC = "End recording"

END_OF_LEARNING = "End of learning"

START_PHASE = "Start phase"
START_PRACTICE_PHASE = "Start practice phase"
END_OF_PHASE = "End of phase"

START_QUESTION = "Start question"
START_SIGNAL = "Start signal"
END_QUESTION = "End question"

START_RESPONSE = "Start response"
END_RESPONSE = "End response"

EXIT = "Exit"

# client modes
LEARN = "Learn"
TEST = "Test"
RECEIVE_PIC = "Receive"
INCOMING_RESPONSE = "Incoming"

# p2p client modes
INIT = "Init"
SPEAKER = "Speaker"
LISTENER = "Listener"
FEEDBACK = "Feedback"
PRACTICE = "Practice"

# p2p server modes
INIT = "Init"
SPEAKERS_TURN = "Speaker's turn"
HEARERS_TURN = "Hearer's turn"
FEEDBACK = "Feedback"

# additional codes for the p2p version
START_ROUND = "Start round"
END_ROUND = "End round"
RESPONSE = "Response"

# Analysis constants
# unit parameters for train_hmm_n_times()
XY, AMP_AND_FREQ, AMP_AND_MEL = "xy", "amp_and_freq", "amp_and_mel"



# GUI Stuff for the new experiment

# Folder for .ui files and .py files generated
# from them
QT_DIR = 'qt_generated'
IMG_DIR = 'img'
MEANING_DIR = join(IMG_DIR, 'meanings', 'featureless')
IMG_EXTENSION = 'png'

FALSE_OVERLAY = join(IMG_DIR, "false.png")
TRUE_OVERLAY = join(IMG_DIR, "true.png")

LEARNING_WIN = 'learningWindow.ui'
INFO_WIN = 'InfoWindow.ui'
TEST_WIN = 'testingWindow.ui'

# Mode constants for Info Window
MOD_PREPHASE = "mod_prephase"
MOD_PRETEST = "mod_pretest"
MOD_EXIT = "mod_exit"
MOD_FIRSTSCREEN = "mod_firstscreen"

# delay between submitting a test answer and 
# the presentation of the next question in msec
DELAY_TEST = 1000

import math

def install_reactor():
    # https://github.com/ghtdak/qtreactor/issues/21
    # from qtreactor import pyside4reactor as reactor
    # from qtreactor import qt4reactor as reactor
    import sys
    if 'qt4reactor' not in sys.modules:
        import qt4reactor
        qt4reactor.install()

def freqToMel(freq):
    return 2595 * math.log10(1 + freq / 700.)


def palmToAmpAndFreq(palmPosition):
    x, y = palmPosition[0], \
        palmPosition[1]

    if x == y == 0:
        return 0, 0
    amp = 1.1 - math.log(abs(y)) / math.log(250.)
    amp = min(1., max(0., amp))
    freq = 110 * (3 ** (abs(x + 200) / 200.))
    return amp, freq


def palmToAmpAndMel(palmPosition):
    amp, freq = palmToAmpAndFreq(palmPosition)
    return amp, freqToMel(freq)


def palmToFreq(palmPosition):
    x, y, z = palmPosition[0], \
        palmPosition[1]

    if x == y == 0:
        return 0, 0

    return 110 * (3 ** (abs(x + 200) / 200.))
