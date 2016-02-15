'''
Created on Feb 21, 2014

@author: kerem
'''

from os.path import expanduser, join, sep

if "ROOT_DIR" not in globals():
    import leaparticulator
    ROOT_DIR = sep.join(leaparticulator.__file__.split(sep)[:-2])

# units
default_pitch = 440.
default_amplitude = .5

leap_port = 9999
# leap_server = "127.0.0.1"
leap_server = "134.184.26.54"
# leap_server = "134.184.26.61"
# leap_server = "127.0.0.1"

fadeout_multiplier = 0.95
fadeout_call_rate = 0.005

# the rate at which we sample the leap info
# for constant rate theremins
THEREMIN_RATE = 1./100

import pyaudio

THEREMIN_AUDIO_FORMAT = pyaudio.paInt32

# maximum duration of a signal in seconds
MAX_SIGNAL_DURATION = 1.

# number of options each test question has
n_of_options = [4, 4, 4]
# number of total available images per phase
n_of_meanings = [5, 10, 15]
# number of test questions per phase
n_of_test_questions = n_of_meanings

question_mark_path = join("img", "question_mark.jpg")

AUDIO_FRAMERATE = 44100
FRAMES_PER_BUFFER = 1024

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
END_SESSION = "End session"

# This is the probability of picking an unestablished meaning for a round
# in P2P. so a random draw greater than this will cause the experiment to
# pick an already established meaning.
NOVELTY_COEFFICENT = 0.55

# Analysis constants
# unit parameters for train_hmm_n_times()
XY, AMP_AND_FREQ, AMP_AND_MEL = "xy", "amp_and_freq", "amp_and_mel"


# GUI Stuff for the new experiment

# Folder for .ui files and .py files generated
# from them
QT_DIR = 'qt_generated'
IMG_DIR = 'img'
MEANING_DIR = join(IMG_DIR, 'meanings', 'featureless')
MEANING_DIR_P2P = join(IMG_DIR, 'meanings', 'featureless')
IMG_EXTENSION = 'png'
P2P_RES_DIR = join(ROOT_DIR, "res", "p2p")
P2P_LOG_DIR = join('logs', 'p2p')

FALSE_OVERLAY = join(IMG_DIR, "false.png")
TRUE_OVERLAY = join(IMG_DIR, "true.png")

LEARNING_WIN = 'learningWindow.ui'
INFO_WIN = 'InfoWindow.ui'
TEST_WIN = 'testingWindow.ui'
P2P_FINAL_WIN = "FinalWindow.ui"

# Mode constants for Info Window
MOD_PREPHASE = "mod_prephase"
MOD_PRETEST = "mod_pretest"
MOD_EXIT = "mod_exit"
MOD_FIRSTSCREEN = "mod_firstscreen"

BROWSER_UI = "BrowserMain.ui"

TXT_EMPTY_SIGNAL = "You have recorded an empty signal. Please record again."

# delay between submitting a test answer and
# the presentation of the next question in msec
DELAY_TEST = 1000

TESTING = False
NO_SOUND = False
RANDOM_SIGNALS = False

import math


def install_reactor():
    # https://github.com/ghtdak/qtreactor/issues/21
    # from qtreactor import pyside4reactor as reactor
    # from qtreactor import qt4reactor as reactor
    import sys
    if 'qt4reactor' in sys.modules:
        print "qt4reactor already in sys.modules!!"
    if 'twisted.internet.reactor' not in sys.modules:
        import qt4reactor
        qt4reactor.install()
        print "Installed qt4reactor"
    else:
        from twisted.internet import reactor
        print "Reactor already installed: %s" % reactor


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


def setupTest(no_sound=False):
    global leap_server, NO_SOUND, TESTING, ROOT_DIR
    leap_server = "127.0.0.1"
    TESTING = True
    ROOT_DIR = expanduser("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt")
    if no_sound:
        NO_SOUND = True

kelly_colors = [
    "#FFB300", # Vivid Yellow
    "#803E75", # Strong Purple
    "#FF6800", # Vivid Orange
    "#A6BDD7", # Very Light Blue
    "#C10020", # Vivid Red
    "#CEA262", # Grayish Yellow
    "#817066", # Medium Gray

    # The following don't work well for people with defective color vision
    "#007D34", # Vivid Green
    "#F6768E", # Strong Purplish Pink
    "#00538A", # Strong Blue
    "#FF7A5C", # Strong Yellowish Pink
    "#53377A", # Strong Violet
    "#FF8E00", # Vivid Orange Yellow
    "#B32851", # Strong Purplish Red
    "#F4C800", # Vivid Greenish Yellow
    "#7F180D", # Strong Reddish Brown
    "#93AA00", # Vivid Yellowish Green
    "#593315", # Deep Yellowish Brown
    "#F13A13", # Vivid Reddish Orange
    "#232C16", # Dark Olive Green
    ]

import matplotlib.colors
kelly_colors = [matplotlib.colors.hex2color(c) for c in kelly_colors]
