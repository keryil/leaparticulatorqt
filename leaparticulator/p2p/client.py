import sys

from PyQt4.QtGui import QApplication

from leaparticulator import constants

print "LeapP2PClient QApp check...",
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print "LeapP2PClient new QApp: %s" % app
else:
    print "LeapP2PClient existing QApp: %s" % app

from leaparticulator.constants import install_reactor

qapplication = app
install_reactor()

import jsonpickle
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.protocols import basic
from twisted.python import log

from leaparticulator.p2p.messaging import InitMessage, LeapP2PMessage, StartMessage, ImageListMessage, \
    StartRoundMessage, ResponseMessage, FeedbackMessage, EndSessionMessage
from leaparticulator.theremin.theremin import Theremin


class LeapP2PClient(basic.LineReceiver):
    delimiter = "\n\n"
    MAX_LENGTH = 1024 * 1024 * 10

    def __init__(self, factory):
        self.factory = factory
        self.ui = self.factory.ui
        self.ui.send_to_server = self.send_to_server
        self.factory.theremin.mute()
        # self.mode = None

    def connectionMade(self):
        self.factory.resetDelay()
        log.msg("Connection made to %s" % self.transport.getPeer())
        self.send_to_server(InitMessage(client_id=self.factory.uid))

    def connectionLost(self, reason):
        log.msg("Connection lost")

    def lineReceived(self, message):
        if len(message) < 1000:
            log.msg("Received: %s" % message)
        else:
            log.msg("Received: %s" % message[:550])

        message = jsonpickle.decode(message)
        assert isinstance(message, LeapP2PMessage)

        if isinstance(message, StartMessage):
            self.factory.mode = constants.INIT
            # self.ui.wait_over()
        elif isinstance(message, ImageListMessage):
            assert self.factory.mode == constants.INIT
            self.factory.mode = constants.IMAGE_LIST
            self.factory.images = message.data
        elif isinstance(message, StartRoundMessage):
            # These (try/catch etc.) are all for debugging
            # remove them ASAP
            # print "ClientFactory mode: ", self.factory.mode
            try:
                assert self.factory.mode in (constants.IMAGE_LIST,
                                             constants.FEEDBACK)
            except AssertionError, err:
                if constants.TESTING:
                    return
                else:
                    raise err
            # self.factory.current_image = message.data.image
            if message.data.isSpeaker:
                log.msg("I am the speaker.")
                self.factory.mode = constants.SPEAKER
                self.factory.theremin.unmute()
                self.ui.wait_over()
                self.factory.current_speaker_image = message.data.image
                self.ui.creation_screen(self.factory.current_speaker_image)
                # self.speak()
                # self.factory.theremin.mute()
            else:
                self.factory.mode = constants.LISTENER
                log.msg("I am the listener. My mode is %s." % self.factory.mode)
                self.factory.theremin.mute()
                self.ui.show_wait()
        elif isinstance(message, ResponseMessage):
            try:
                assert self.factory.mode == constants.LISTENER
            except AssertionError:
                raise Exception("Mode not LISTENER: %s" % self.factory.mode)
            self.factory.theremin.mute()
            self.factory.last_response_data = message.data

            options = message.data.options
            log.msg("Received options: %s" % options)
            self.ui.wait_over()
            self.ui.test_screen(options)
            # self.listen()
        elif isinstance(message, FeedbackMessage):
            assert self.factory.mode == constants.FEEDBACK
            print "Received feedback: %s" % message
            self.ui.wait_over()
            self.factory.image_pointer = message.image_pointer
            self.ui.feedback_screen(message.target_image,
                                    message.chosen_image)
            # self.send_to_server(EndRoundMessage())
        elif isinstance(message, EndSessionMessage):
            self.ui.final_screen()

    def send_to_server(self, message):
        assert isinstance(message, LeapP2PMessage)
        self.sendLine(jsonpickle.encode(message))

    def speak(self):
        # from time import sleep
        # sleep(2)
        message = ResponseMessage(signal=self.ui.getSignal(),
                                  image=self.factory.current_speaker_image)
        self.send_to_server(message)
        self.ui.resetSignal()
        # self.factory.last_signal = []
        print "Spoken"
        self.factory.mode = constants.FEEDBACK

    def listen(self, image):
        print "Listening"
        self.factory.current_hearer_image = image
        # print self.factory.last_response_data
        self.send_to_server(ResponseMessage(
                signal=self.factory.last_response_data.signal,
                image=image,  # choice(self.factory.images))
                options=self.factory.last_response_data.options))
        self.factory.mode = constants.FEEDBACK
        # def extend_last_signal(self,signal):
        #     print "Extending: %s" % signal


class LeapP2PClientFactory(protocol.ReconnectingClientFactory):
    protocol = LeapP2PClient
    last_response_data = None
    recording = False

    def __init__(self, leap_listener, ui, uid):
        log.startLogging(sys.stdout)
        if ui is None:
            log.msg("Warning: No UI")
        self.theremin = leap_listener
        self.ui = ui
        self.uid = uid
        # self.phase = 0
        self.__mode = None
        self.image_pointer = 2
        self.mode

    def buildProtocol(self, addr):
        # self.listener = leap_listener
        c = LeapP2PClient(self)
        log.msg("New LeapP2PClient initialized")
        return c

    def getSignal(self):
        log.msg("The recorded signal is %d frames long" %
                len(self.theremin.get_signal()))
        return self.theremin.last_signal
        # print self.last_signal

    @property
    def mode(self):
        log.msg("SOMEONE ASKED MY MODE! IT IS %s" % self.__mode)
        return self.__mode

    @mode.setter
    def mode(self, value):
        assert value in (constants.INIT,
                         constants.SPEAKER,
                         constants.LISTENER,
                         constants.FEEDBACK,
                         constants.PRACTICE)
        log.msg("MY MODE IS BEING SET TO %s" % value)
        self.__mode = value


def start_client(qapplication, uid):
    assert uid is not None
    from leaparticulator.p2p.ui.client import LeapP2PClientUI

    print "Init UI object..."
    ui = LeapP2PClientUI(qapplication)
    print "Init theremin..."
    # theremin, reactor, controller, call = gimmeSimpleTheremin(n_of_notes=1,
    #                                                               # Constants.default_amplitude,
    #                                                               default_volume=None,
    #                                                               ui=ui, realtime=True)
    # theremin, reactor, controller, connection = gimmeSomeTheremin(n_of_notes=1,
    #                                                               # Constants.default_amplitude,
    #                                                               default_volume=None,
    #                                                               ui=ui, realtime=False,
    #                                                               factory=None,
    #                                                               ip=None)
    theremin = Theremin(ui=ui, realtime=False, factory=None)
    factory = LeapP2PClientFactory(theremin, ui=ui, uid=uid)
    theremin.factory = factory
    # theremin.call = call
    theremin.reactor = reactor
    theremin.player.ui = ui
    print "Initiating connection with %s:%s" % (constants.leap_server, constants.leap_port)
    endpoint = TCP4ClientEndpoint(
            reactor, constants.leap_server, constants.leap_port)
    theremin.factory = factory
    factory.theremin = theremin
    # ui.setClientFactory(factory)
    theremin.endpoint = endpoint

    def go(client):
        ui.setClient(client)
        if not (constants.TESTING or reactor.running):
            print "Starting reactor..."
            reactor.runReturn()
            print "Starting UI..."
            ui.go()

    connection_def = endpoint.connect(factory)
    connection_def.addCallback(go)
    factory.connection_def = connection_def
    factory.endpoint = endpoint

    connection = None
    return theremin
