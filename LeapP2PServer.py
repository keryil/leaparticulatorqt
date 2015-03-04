# import LeapServer

once = False
qapplication = None
import sys
from Constants import install_reactor
from PySide.QtGui import QApplication
qapplication = QApplication(sys.argv)
install_reactor()
# import sys, platform
# if "qt4reactor" not in sys.modules:
#     from PySide.QtGui import QApplication
#     qapplication = QApplication(sys.argv)
#     if platform.system() == "Darwin":
#         from twisted.internet import cfreactor
#         cfreactor.install()    
#     else:   
#         import qt4reactor
#         qt4reactor.install()
#         # from twisted.internet import gtk2reactor
#         # gtk2reactor.install()
#         once = True

from twisted.internet import protocol, reactor
from twisted.internet.endpoints import TCP4ServerEndpoint, TCP4ClientEndpoint
from twisted.internet.protocol import ClientFactory
import Leap, sys
from twisted.python import log
from twisted.python.log import FileLogObserver
from twisted.protocols import basic
from datetime import datetime
from random import choice, sample, shuffle
import Constants
import jsonpickle
import TestQuestion
from LeapFrame import LeapFrame
from P2PMessaging import *
from LeapTheremin import gimmeSomeTheremin, ThereminPlayback


# class LeapP2PRoundData(object):
#     hearer = None
#     speaker = None
#     speakers_response_data = None
#     hearers_response_data = None
#     test_images = []


class LeapP2PRoundSummary(object):

    def __init__(self):
        self.speaker = None
        self.hearer = None
        self.signal = None
        self.image = None
        self.guess = None
        self.success = None

    def set_participants(self, speaker, hearer):
        self.speaker = speaker
        self.hearer = hearer

    def set_speakers_contribution(self, response_message):
        self.image = response_message.data.image
        self.signal = response_message.data.signal

    def set_hearers_contribution(self, response_message):
        self.guess = response_message.data.image
        self.success = (self.guess == self.image)


def notifies(f):
    def notifies_(arg):
        def notifier(self, *args):
            f(self, *args)
            getattr(self, notify)()
        return notifier
    return notifies_

class LeapP2PSession(object):
    # round_data = []
    # participants = []
    def __init__(self, participants, condition):
        self.round_data = []
        self.participants = list(participants)
        self.callbacks = []
        self.condition = condition

    def newRound(self):
        if len(self.round_data) > 1:
            rnd = self.getLastRound()
            assert None not in (rnd.speaker, rnd.hearer, rnd.signal,
                                rnd.image, rnd.guess, rnd.success)
        self.round_data.append(LeapP2PRoundSummary())
        self.notify()

    def setParticipants(self, speaker, hearer):
        self.getLastRound().set_participants(speaker, hearer)
        self.notify()

    def setSpeakerContribution(self, response_message):
        self.getLastRound().set_speakers_contribution(response_message)
        self.notify()

    def setHearerContribution(self, response_message):
        self.getLastRound().set_hearers_contribution(response_message)
        self.notify()

    def getLastRound(self):
        return self.round_data[-1]

    def getHearer(self):
        return self.getLastRound().hearer

    def getSpeaker(self):
        return self.getLastRound().speaker        

    def addCallback(self, func):
        self.callbacks.append(func)

    def clearCallbacks(self):
        self.callbacks = []

    def notify(self):
        for func in self.callbacks:
            func(self)


class LeapP2PServer(basic.LineReceiver):
    """
    An extension of the regular server that supports runs with multiple
    peers communicating with one another. 
    """
    other_end = ""
    other_end_alias = None
    delimiter = "\n\n"
    MAX_LENGTH = 1024*1024*10
    response = []
    phase = 0
    image_mask_1 = "./img/meanings/5_*.png"
    image_mask_2 = "./img/meanings/*_[135].png"
    image_mask_3 = "./img/meanings/*_[135].png"
    image_mask = "./img/meanings/[135]_[12345].png"
    recording = False
    n_of_test_questions = [5,9,9]
    n_of_options = [4,4,4]
    end_round_msg_counter = 0

    # callbacks
    connectionMadeListeners = set()
    connectionLostListeners = set()

    def __init__(self, factory):  # , image_mask="./img/animals/*.png"):
        self.factory = factory
        # self.factory.mode = Constants.INIT
        if len(self.factory.images[0]) == 0:
            from glob import glob
            from random import shuffle
            log.msg("Found image files: %s" % glob(self.image_mask))
            # self.factory.images[0] = glob(self.image_mask_1)
            # self.factory.images[1] = glob(self.image_mask_2)
            # self.factory.images[2] = glob(self.image_mask_3)
            # shuffle(self.factory.images[0])
            # shuffle(self.factory.images[1])
            # shuffle(self.factory.images[2])
            self.factory.images = glob(self.image_mask)
            shuffle(self.factory.images)
            print "Images in place!"
            log.msg(self.factory.images)

    def addListenerConnectionLost(self, f):
        self.connectionLostListeners.add(f)

    def addListenerConnectionMade(self, f):
        self.connectionMadeListeners.add(f)

    def connectionMade(self):
        self.other_end = self.transport.getPeer().host
        self.factory.clients[self] = self.other_end
        # check if we need a new session
        # if self.other_end not in self.factory.responses:
        #     self.factory.responses[self.other_end] = {0:{},1:{},2:{}}
        #     self.factory.responses_practice[self.other_end] = {0:{},1:{},2:{}}
        log.msg("Connection with %s is made" % self.other_end)
        # log.msg("Callbacks: %s" % self.connectionMadeListeners)
        for callback in self.connectionMadeListeners:
            callback(self)

    def connectionLost(self, reason):
        del self.factory.clients[self]
        for callback in self.connectionLostListeners:
            callback(self, reason)
        log.msg("Connection with %s (%s) is lost because: %s" % (self.other_end, self.other_end_alias, reason))

    def send_all(self, message):
        """
        Send a message to all connected clients.
        """
        for c in self.factory.clients.keys():
            c.send_to_client(message, c)

    def send_to_client(self, message, client):
        """
        Send a message to a specific client.
        """
        assert isinstance(message, LeapP2PMessage)
        assert client in self.factory.clients.keys()
        message = jsonpickle.encode(message)
        nline = "<{}@{}> {}".format(self.factory.clients[client], client.transport.getPeer().host, message[:100])
        log.msg("Sending %s" % nline)
        client.sendLine(message)

    def start(self, practice = False):
        log.msg("Starting a new session with clients %s" % ([c for c in self.factory.clients.values()]))
        self.factory.practice = practice
        self.factory.ui.disableStart()
        # self.factory.rounds.append(LeapP2PRoundSummary())
        self.factory.end_round_msg_counter = 0
        # make sure we have exactly two clients
        if len(self.factory.clients) == 2:
            # send the start signals and the image list
            if self.factory.mode == Constants.INIT:
                self.factory.session = LeapP2PSession(self.factory.clients,
                                                      self.factory.condition)
                self.factory.session.addCallback(self.factory.ui.onSessionChange)
                # self.factory.session_data = LeapP2PSession(self.factory.clients)
                self.send_all(StartMessage())
                self.send_all(ImageListMessage(self.factory.images))
            self.choose_speaker_and_topic()

    def choose_speaker_and_topic(self):
        # choose a speaker 
        log.msg("Choosing the speaker and topic...")
        # TODO: how to choose the speaker?
        self.factory.mode = Constants.SPEAKERS_TURN
        # speaker_no = choice(range(2))
        hearer = speaker = choice(tuple(self.factory.clients.keys()))
        while hearer == speaker:
            hearer = choice(tuple(self.factory.clients.keys()))
        assert hearer != speaker

        # choose an image
        # TODO: we need a strategy for this
        image = choice(self.factory.images)
        log.msg("The chosen image is: %s" % image)
        log.msg("Speaker: %s; Hearer: %s" % (self.factory.clients[speaker], 
                                             self.factory.clients[hearer]))
        self.factory.session.newRound()
        self.factory.session.getLastRound().image = image
        self.factory.session.setParticipants(speaker=speaker,
                                                 hearer=hearer)
        # send the messages
        self.send_to_client(StartRoundMessage(isSpeaker=True,image=image), speaker)
        self.send_to_client(StartRoundMessage(isSpeaker=False,image=image), hearer)

    def lineReceived(self, line):
        nline = "<{}@{}> {}".format(self.other_end_alias, self.other_end, line)
        if len(nline) < 300: 
            log.msg("Received: %s" % nline)
        message = jsonpickle.decode(line)
        assert isinstance(message, LeapP2PMessage)

        if self.factory.mode == Constants.INIT:
            assert isinstance(message, InitMessage)
            self.factory.clients[self] = message.client_id
            self.other_end_alias = message.client_id
            self.factory.ui.connectionMade(self.other_end, message.client_id)
            if len(self.factory.clients) < 2:
                return
            else:
                all_aliased = True
                for c in self.factory.clients:
                    if c.other_end_alias == "":
                        all_aliased = False
                        break
                if all_aliased:
                    self.factory.ui.btnStart.clicked.connect(self.start)
                    self.factory.ui.enableStart()

        elif self.factory.mode == Constants.SPEAKERS_TURN:
            assert isinstance(message, ResponseMessage)
            # print "Received signal: %s" % message.data.signal[-5:]
            self.factory.mode = Constants.HEARERS_TURN
            self.factory.session.setSpeakerContribution(message)
            self.send_to_client(message, 
                                self.factory.session.getHearer())
        elif self.factory.mode == Constants.HEARERS_TURN:
            assert isinstance(message, ResponseMessage)
            self.factory.session.setHearerContribution(message)
            # give feedback
            self.factory.mode = Constants.FEEDBACK
            self.send_all(FeedbackMessage(target_image=self.factory.session.getLastRound().image,
                                          chosen_image=self.factory.session.getLastRound().guess,
                                          success=self.factory.session.getLastRound().success))
        elif self.factory.mode == Constants.FEEDBACK:
            assert isinstance(message, EndRoundMessage)
            self.factory.end_round_msg_counter += 1
            if self.factory.end_round_msg_counter == len(self.factory.clients):
                if len(self.factory.session.round_data) < 20:
                    self.start()
            else:
                log.msg("%i EndRoundMessages received so far." % self.factory.end_round_msg_counter)

    

class LeapP2PServerFactory(protocol.Factory):
    numConnections = 0
    protocol = LeapP2PServer
    # Mode is either play or listen
    mode = Constants.LEARN
    image_index = 0
    images = [[], [], []]
    questions_by_phase = [[], [], []]
    rounds = []
    ui = None
    uid = None
    practice = False
    session_data = None
    clients = {}
    end_round_msg_counter = 0

    # dict of dicts e.g. responses[client][phase][image] = response
    # responses = {}
    # responses_practice = {}
    # # dict of dicts e.g. responses[client][phase] = [questions]
    # test_results = {}
    # test_results_practice = {}
    condition = None

    def __init__(self,ui=None, condition=None, no_log=False, uid=None):
        import time
        self.clients = {}
        self.ui = ui
        self.session = None
        if uid is None:
            self.uid = time.strftime("%y%m%d.%H%M%S")
        else:
            self.uid = uid
        conditions = ["1","2","1r","2r"]
        if condition not in conditions:
            raise Exception("Invalid condition %s. Should be one of %s" % (condition, conditions))
        self.condition = condition
        self.mode = Constants.INIT
        log.startLogging(sys.stdout)
        if not no_log:
            log.addObserver(FileLogObserver(open("logs/%s.log" % self.uid,'w')).emit)
        log.msg("Condition: %s" % condition)

    def buildProtocol(self, addr):
        server = LeapP2PServer(self)
        LeapP2PServer.MAX_LENGTH = 100000 * 1024
        LeapP2PServer.delimiter = "\n\n"
        server.delimiter = "\n\n"
        server.MAX_LENGTH = 100000 * 1024
        if self.ui:
            server.addListenerConnectionLost(self.ui.connectionLost)
        #     server.addListenerConnectionMade(self.ui.connectionMade)
        return server

class LeapP2PClient(basic.LineReceiver):
    delimiter = "\n\n"
    MAX_LENGTH = 1024*1024*10
    def __init__(self, factory):
        self.factory = factory
        self.ui = self.factory.ui
        self.factory.theremin.mute()
        # self.ui.show_wait()
        # self.ui.go()

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
            log.msg("Received: %s" % message[:250])
            
        message = jsonpickle.decode(message)
        assert isinstance(message, LeapP2PMessage)

        if isinstance(message, StartMessage):
            self.factory.mode = Constants.INIT
            # self.ui.wait_over()
        elif isinstance(message, ImageListMessage):
            self.factory.mode = Constants.IMAGE_LIST
            self.factory.images = message.data
        elif isinstance(message, StartRoundMessage):
            assert self.factory.mode in (Constants.IMAGE_LIST, 
                                         Constants.FEEDBACK)
            self.factory.current_image = message.data.image
            if message.data.isSpeaker:
                self.factory.mode = Constants.SPEAKER
                self.factory.theremin.unmute()
                self.ui.wait_over()
                self.ui.creation_screen(self.factory.current_image)
                # self.speak()
                # self.factory.theremin.mute()
            else:
                self.factory.mode = Constants.LISTENER
                self.ui.show_wait()
        elif isinstance(message, ResponseMessage):
            assert self.factory.mode == Constants.LISTENER
            self.factory.theremin.mute()
            self.factory.last_response_data = message.data
            options = sample(list(set(self.factory.images) - set([message.data.image])),3) + [message.data.image]
            shuffle(options)
            self.ui.wait_over()
            self.ui.test_screen(options)
            # self.listen()
        elif isinstance(message, FeedbackMessage):
            assert self.factory.mode == Constants.FEEDBACK
            print "Received feedback: %s" % message
            self.ui.wait_over()
            self.ui.feedback_screen(message.target_image, 
                                    message.chosen_image)
            # self.send_to_server(EndRoundMessage())

    def send_to_server(self, message):
        assert isinstance(message, LeapP2PMessage)
        self.sendLine(jsonpickle.encode(message))

    def speak(self):
        # from time import sleep
        # sleep(2)
        message = ResponseMessage(signal=self.ui.getSignal(),
                                  image=self.factory.current_image)
        self.send_to_server(message)
        self.ui.resetSignal()
        # self.factory.last_signal = []
        print "Spoken"
        self.factory.mode = Constants.FEEDBACK

    def listen(self, image):
        print "Listening"
        # print self.factory.last_response_data
        from random import choice
        self.send_to_server(ResponseMessage(
                            signal=self.factory.last_response_data.signal,
                            image=image)#choice(self.factory.images))
                            )
        self.factory.mode = Constants.FEEDBACK
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
    
    # def start_recording(self):
    #     self.resetSignal()
    #     self.theremin.record()
    #     log.msg("start_recording()")
    #     # import traceback
    #     # traceback.print_stack()

    # def stop_recording(self):
    #     log.msg("stop_recording() at %d frames" % len(self.theremin.get_signal()))
    #     self.theremin.stop_record()

    # def resetSignal(self):
    #     log.msg("resetSignal()")
    #     self.theremin.reset_signal()

    def buildProtocol(self, addr):
        # self.listener = leap_listener        
        c = LeapP2PClient(self)
        # TODO: I assign none for the callback because we don't need realtime signals,
        # we need response-by-response playback
        # self.listener.protocol = None
        log.msg("New LeapP2PClient initialized")
        return c

    # def extendSignal(self, frame):
    #     # log.msg("Call to extendSignal()")
    #     if self.recording:
    #         self.last_signal.append(frame)
    #         log.msg("Recorded new frame (#%d)..." % len(self.last_signal))

    def getSignal(self):
        log.msg("The recorded signal is %d frames long" % len(self.theremin.get_signal()))
        return self.theremin.last_signal
        # print self.last_signal

def get_server_instance(condition, ui=None):
    """
    Returns a default server instance, reading settings 
    from Constants.py
    """
    endpoint = TCP4ServerEndpoint(reactor, Constants.leap_port)
    endpoint.listen(LeapP2PServerFactory(ui=ui,condition=condition, no_log=True))
    return endpoint


if __name__ == '__main__':
    import sys
    from LeapP2PClientUI import LeapP2PClientUI
    from LeapP2PServerUI import LeapP2PServerUI
    try:
        assert len(sys.argv) > 2
        if sys.argv[2] == "client":
            assert len(sys.argv) > 3
            ui = LeapP2PClientUI(qapplication)
            theremin, reactor, controller, connection = gimmeSomeTheremin(n_of_notes=1, 
                                                                          default_volume=None,#Constants.default_amplitude, 
                                                                          ui=ui, realtime=False, 
                                                                          factory=LeapP2PClientFactory,
                                                                          ip = None)
            endpoint = TCP4ClientEndpoint(reactor, Constants.leap_server, Constants.leap_port)
            factory = LeapP2PClientFactory(theremin, ui=ui, uid=sys.argv[3])
            theremin.factory = factory
            # ui.setClientFactory(factory)
            connection_def = endpoint.connect(factory)
            connection_def.addCallback(ui.setClient)
            # ui.setClient(connection)
            reactor.runReturn()
            ui.go()
        elif sys.argv[2] == "server":
            try:
                ui = LeapP2PServerUI(qapplication)
                get_server_instance(condition=sys.argv[1], ui=ui)
                reactor.runReturn()
                ui.go()
            except IndexError, e:
                import traceback
                traceback.print_exc()
                print "ERROR: You should specify a condition (1/2/1r/2r) as a command line argument."
                sys.exit(-1)
    except AssertionError:
                print "USAGE: LeapP2PServer {condition} {client/server} [client_id]."
                sys.exit(-1)