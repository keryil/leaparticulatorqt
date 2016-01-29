# import LeapServer
from leaparticulator.p2p.client import start_client

once = False
qapplication = None

import sys

from PyQt4.QtGui import QApplication

print "LeapP2PServer QApp check...",
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print "LeapP2PServer new QApp: %s" % app
else:
    print "LeapP2PServer existing QApp: %s" % app

from leaparticulator.constants import install_reactor, NOVELTY_COEFFICENT

qapplication = app
install_reactor()

from twisted.internet import protocol, reactor
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.python import log
from twisted.python.log import FileLogObserver
from twisted.protocols import basic
from random import choice, sample, shuffle
import jsonpickle
from leaparticulator.p2p.messaging import *
from leaparticulator.data.meaning import P2PMeaning
import os


class LeapP2PRoundSummary(object):

    def __init__(self):
        self.speaker = None
        self.hearer = None
        self.signal = None
        self.image = None
        self.guess = None
        self.success = None
        self.options = None
        # these are the counts at the end of the round.
        self.success_counts = None

    def set_participants(self, speaker, hearer):
        self.speaker = speaker
        self.hearer = hearer

    def set_speakers_contribution(self, response_message):
        self.image = response_message.data.image
        self.signal = response_message.data.signal

    def set_hearers_contribution(self, response_message):
        self.guess = response_message.data.image
        self.success = (str(self.guess) == str(self.image))

    def set_options(self, options):
        assert options is not None
        self.options = options

    def set_success_counts(self, counts):
        from copy import deepcopy
        self.success_counts = deepcopy(counts)

    def shorthand_copy(self):
        """
        Returns a copy of this object with the TCP connection objects
        replaced by strings alias@ip-address for use in log files
        where we don't need the full object at all.
        """
        extract_participant = lambda x: "%s@%s" % (x.other_end_alias, x.other_end)
        copy = LeapP2PRoundSummary()
        copy.speaker = extract_participant(self.speaker)
        copy.hearer = extract_participant(self.hearer)
        copy.signal = self.signal
        copy.image = self.image
        copy.options = self.options
        copy.guess = self.guess
        copy.success = self.success
        copy.success_counts = self.success_counts
        return copy

class LeapP2PSession(object):
    def __init__(self, participants, condition, factory):
        self.round_data = []
        self.participants = list(participants)
        self.callbacks = []
        self.condition = condition
        self.factory = factory
        self.started_file_dump = False

    def dump_to_file(self):
        from os.path import join
        from jsonpickle import encode
        filename = "P2P-%s.%s.exp.log" % (self.factory.uid, self.factory.condition)
        exists = self.started_file_dump
        filename = join(constants.ROOT_DIR, "logs", filename)

        print "Dumping round data to %s. First entry? %s" % (filename, not exists)
        extract_participant = lambda x: "%s@%s" % (x.other_end_alias, x.other_end)
        mode = "a" if exists else "w"
        with open(filename, mode) as f:
            if not exists:
                f.write(encode([extract_participant(p) for p in self.participants]))
                f.write("\n")
                f.write(encode(self.factory.images))
                f.write("\n")
            f.write(encode(self.getLastRound().shorthand_copy()))
            f.write("\n")
        self.started_file_dump = True


    def newRound(self):
        if len(self.round_data) > 0:
            rnd = self.getLastRound()
            assert rnd.success is not None
            counts = rnd.success_counts
        #     assert None not in (rnd.speaker, rnd.hearer, rnd.signal,
        #                         rnd.image, rnd.guess, rnd.success)
        else:
            counts = self.factory.image_success
        log.msg("New round: #%d" % len(self.round_data))
        # from traceback import print_stack; print_stack()
        self.round_data.append(LeapP2PRoundSummary())
        self.setSuccessCounts(counts)
        self.notify()

    def setOptions(self, options):
        self.getLastRound().set_options(options)
        self.notify()

    def setImage(self, image):
        self.image = image

    def setParticipants(self, speaker, hearer):
        self.getLastRound().set_participants(speaker, hearer)
        self.notify()

    def setSpeakerContribution(self, response_message):
        self.getLastRound().set_speakers_contribution(response_message)
        self.notify()

    def setHearerContribution(self, response_message):
        self.getLastRound().set_hearers_contribution(response_message)
        # since this is the last piece of information, we can
        # serialize after setting the hearer contribution
        self.dump_to_file()
        self.notify()

    def getLastRound(self):
        return self.round_data[-1]

    def getHearer(self):
        return self.getLastRound().hearer

    def getSpeaker(self):
        return self.getLastRound().speaker

    def setSuccessCounts(self, counts):
        self.getLastRound().set_success_counts(counts)
        self.notify()

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
    MAX_LENGTH = 1024 * 1024 * 10
    response = []
    # phase = 0

    image_mask = os.path.join(constants.MEANING_DIR_P2P, "*.%s" %
                              constants.IMG_EXTENSION)
    recording = False
    # n_of_test_questions = [5, 9, 9]
    # n_of_options = [4, 4, 4]

    # callbacks
    connectionMadeListeners = set()
    connectionLostListeners = set()

    def __init__(self, factory):  # , image_mask="./img/animals/*.png"):
        self.factory = factory
        self.end_round_msg_counter = -1
        # self.factory.mode = Constants.INIT
        if len(self.factory.images) == 0:
            from glob import glob
            from random import shuffle
            # root = os.getcwd()
            # if "_trial_temp" in root:
            #     root = root[:-11]
            log.msg("Found image files: %s" % glob(os.path.join(constants.ROOT_DIR,
                                                self.image_mask)))

            images = glob(os.path.join(constants.ROOT_DIR, self.image_mask))
            self.factory.images = [P2PMeaning.FromFile(i) for i in images]
            shuffle(self.factory.images)
            # this keeps the pointer so that the images in the game are
            # self.factory.images[:self.factory.image_pointer] where
            # self.factory.image_success[image] gives the number of correct
            # guesses regarding image.

            self.factory.image_pointer = 2
            self.factory.image_success = {str(image): 0 for image in self.factory.images}
            print "Success dict, initialized: %s" % self.factory.image_success
            # print self.factory.image_success
            # print images[0], type(images[0])
            print "Images in place!"
            log.msg(self.factory.images)

    def addListenerConnectionLost(self, f):
        self.connectionLostListeners.add(f)

    def addListenerConnectionMade(self, f):
        self.connectionMadeListeners.add(f)

    def connectionMade(self):
        self.other_end = self.transport.getPeer().host

        # bugfix for multiple connections to hosts
        # no longer needed because we stopped the theremin
        # establishing a second connection

        # for k in self.factory.clients.keys():
        #     print "Checking %s versus %s" % (self.other_end, k.transport.getPeer().host)
        #     # skip if this is a server/client. we are probably testing.
        #     if self.other_end == constants.leap_server:
        #         continue
        #     if k.transport.getPeer().host == self.other_end:
        #         del self.factory.clients[k]

        self.factory.clients[self] = self.other_end

        # check if we need a new session
        # if self.other_end not in self.factory.responses:
        #     self.factory.responses[self.other_end] = {0:{},1:{},2:{}}
        #     self.factory.responses_practice[self.other_end] = {0:{},1:{},2:{}}
        log.msg("Connection with %s is made" % self.other_end)
        print "Current client list:", self.factory.clients
        # log.msg("Callbacks: %s" % self.connectionMadeListeners)
        for callback in self.connectionMadeListeners:
            callback(self)

    def connectionLost(self, reason):
        del self.factory.clients[self]
        for callback in self.connectionLostListeners:
            callback(self, reason)
        log.msg("Connection with %s (%s) is lost because: %s" %
                (self.other_end, self.other_end_alias, reason))

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
        nline = "<{}@{}> {}".format(
            self.factory.clients[client], client.transport.getPeer().host, message[:300])
        log.msg("Sending %s" % nline)
        client.sendLine(message)

    def end(self):
        """
        End the experiment.
        """
        self.send_all(EndSessionMessage())

    def start(self, practice=False):
        if self.factory.mode == constants.INIT:
            print "--------->start() called"
            print "clients (%d): %s" % (len(self.factory.clients), self.factory.clients)

        self.factory.practice = practice
        self.factory.ui.disableStart()
        self.factory.ui.enableEnd()
        # self.factory.rounds.append(LeapP2PRoundSummary())
        # make sure we have exactly two clients
        if len(self.factory.clients) == 2:

            if self.factory.end_round_msg_counter not in (-1, 2):
                print 'Why the fuck are we restarting with %d end of round messages?! Ignoring...' % self.factory.end_round_msg_counter
                return

            # check if we have any images left, otherwise, terminate
            if self.factory.image_pointer == len(self.factory.images):
                log.msg("Successfully finished the image set, ending the experiment...")
                self.end()
                return

            self.factory.end_round_msg_counter = 0
            log.msg("Starting a new session with clients %s" %
                    ([c for c in self.factory.clients.values()]))
            # send the start signals and the image list
            if self.factory.mode == constants.INIT:
                self.factory.session = LeapP2PSession(self.factory.clients,
                                                      self.factory.condition,
                                                      self.factory)
                self.factory.session.addCallback(
                    self.factory.ui.onSessionChange)

                # self.factory.session_data = LeapP2PSession(self.factory.clients)
                self.send_all(StartMessage())
                img_list = ImageListMessage(self.factory.images)
                # print self.factory.images
                # print img_list
                self.send_all(img_list)
            self.choose_speaker_and_topic()
        else:
            print "--------->start() called with more or less than two clients"


    def choose_speaker_and_topic(self):
        # choose a speaker
        log.msg("Choosing the speaker and topic...")
        # choose the speaker and listener alternately

        self.factory.mode = constants.SPEAKERS_TURN
        # speaker_no = choice(range(2))
        hearer = speaker = None
        if len(self.factory.clients) == 2 and \
                        len(self.factory.session.round_data) > 0:
            last_round = self.factory.session.getLastRound()
            hearer = last_round.speaker
            speaker = last_round.hearer
        else:
            hearer = speaker = choice(tuple(self.factory.clients.keys()))
            while hearer == speaker:
                hearer = choice(tuple(self.factory.clients.keys()))
        assert hearer != speaker

        # choose an image

        # We have n images, x of which have fewer than 2 correct responses.
        # Let's give 66% of the probability to the responses without 2 correct responses, so each of them gets 1/2x.
        # Then the other meanings get 1/2(n-x) each.

        # this is a flag indicating whether or not
        # we are going to pick an image with 2
        # consecutive right guesses, at p = .5
        from random import random
        pick_guessed_image = True if random() > NOVELTY_COEFFICENT else False
        log.msg("Do we intend to pick an already established meaning? %s" % pick_guessed_image)

        all_images = self.factory.images[:self.factory.image_pointer]
        success = self.factory.image_success
        guessed = filter(lambda x: success[str(x)] == 2, all_images)
        non_guessed = filter(lambda x: success[str(x)] != 2, all_images)

        if pick_guessed_image:
            log.msg("Do we have enough established meanings to pick from? %s" % (len(guessed) > 1))

        if pick_guessed_image and len(guessed) > 1:
            image = choice(guessed)
        else:
            image = choice(non_guessed)

        log.msg("The chosen image is: %s" % image)
        log.msg("Speaker: %s; Hearer: %s" % (self.factory.clients[speaker],
                                             self.factory.clients[hearer]))
        print "Success dict:", success
        print "Image pointer:", self.factory.image_pointer
        self.factory.session.newRound()
        self.factory.session.getLastRound().image = image
        self.factory.session.setParticipants(speaker=speaker,
                                             hearer=hearer)
        # send the messages
        self.send_to_client(
            StartRoundMessage(isSpeaker=True, image=image), speaker)
        self.send_to_client(
            StartRoundMessage(isSpeaker=False, image=image), hearer)

    def expandMeaningSpace(self):
        """
        Checks whether or not it's time to expand the meaning space,
        and does so if necessary.
        :return:
        """
        # check that the *whole* meaning space is
        # figured out before expanding it
        for img in self.factory.images[:self.factory.image_pointer]:
            count = self.factory.image_success[str(img)]
            if count < 2:
                log.msg("Not expanding meaning space due to %s (%s correct guesses so far)" % (img, count))
                break
        else:
            self.factory.image_pointer += 2
            # if we run out of images, just end the experiment.
            if self.factory.image_pointer >= len(self.factory.images):
                self.end()
            else:
                self.factory.image_pointer = min(self.factory.image_pointer,
                                                 len(self.factory.images))
                log.msg("Expanding meaning space by two; the new space is\n%s" % (
                    self.factory.images[:self.factory.image_pointer]))

    def lineReceived(self, line):
        nline = "<{}@{}> {}".format(self.other_end_alias, self.other_end, line)
        if len(nline) < 300:
            log.msg("Received: %s" % nline)
        else:
            log.msg("Received: %s" % nline[:400])
        message = jsonpickle.decode(line)
        assert isinstance(message, LeapP2PMessage)

        if self.factory.mode == constants.INIT:
            if not hasattr(self.factory, 'start_counter'):
                log.msg("Initialized start counter at zero.")
                self.factory.start_counter = 0

            if isinstance(message, InitMessage):
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
            else:
                assert isinstance(message, StartMessage)
                self.factory.start_counter += 1
                print "Received START message #%d" % self.factory.start_counter
                if self.factory.start_counter == 2:
                    # self.factory.ui.btnStart.clicked.connect(self.start)
                    # self.factory.ui.enableStart()
                    del self.factory.start_counter
                    self.start()

        elif self.factory.mode == constants.SPEAKERS_TURN:
            assert isinstance(message, ResponseMessage)
            # print "Received signal: %s" % message.data.signal[-5:]
            self.factory.mode = constants.HEARERS_TURN
            self.factory.session.setSpeakerContribution(message)

            # pick the images
            image_pointer = self.factory.image_pointer
            options = filter(lambda x: str(x) != str(message.data.image), self.factory.images[:image_pointer])
            if len(options) > 3:
                options = sample(options, 3)
            options = options + [message.data.image]
            shuffle(options)
            assert len(map(str, options)) == len(set(map(str, options)))
            log.msg("Prepared options for the round: %s" % options)
            self.factory.session.setOptions(options)

            message.data.options = options

            self.send_to_client(message,
                                self.factory.session.getHearer())
        elif self.factory.mode == constants.HEARERS_TURN:
            assert isinstance(message, ResponseMessage)
            self.factory.session.setHearerContribution(message)
            # give feedback
            success = self.factory.session.getLastRound().success
            image = self.factory.session.getLastRound().image
            guess = self.factory.session.getLastRound().guess
            self.factory.mode = constants.FEEDBACK

            if success:
                if self.factory.image_success[str(image)] < 2:
                    try:
                        self.factory.image_success[str(image)] += 1
                    except Exception, e:
                        print ["\"%s\"" % i for i in self.factory.image_success.keys()]
                        print "\"%s\"" % image
                        print e
                        raise Exception()
                self.expandMeaningSpace()
            else:
                # we only expand the meaning space if there are two **consecutive** successes
                # so we reset the count for any failure
                self.factory.image_success[str(image)] = 0
            self.factory.session.setSuccessCounts(self.factory.image_success)

            self.send_all(FeedbackMessage(target_image=image,
                                          chosen_image=guess,
                                          success=success,
                                          image_pointer=self.factory.image_pointer)
                          )


        elif self.factory.mode == constants.FEEDBACK:
            assert isinstance(message, EndRoundMessage)
            self.factory.end_round_msg_counter += 1
            log.msg("%i EndRoundMessages received so far." %
                    self.factory.end_round_msg_counter)
            if self.factory.end_round_msg_counter == len(self.factory.clients):
                log.msg("Moving to the next round...")
                self.start()

class LeapP2PServerFactory(protocol.Factory):
    numConnections = 0
    protocol = LeapP2PServer
    # Mode is either play or listen
    mode = constants.LEARN
    image_index = 0
    images = []
    # questions_by_phase = [[], [], []]
    rounds = []
    ui = None
    uid = None
    practice = False
    session_data = None
    clients = {}
    end_round_msg_counter = -1
    logger = log

    # dict of dicts e.g. responses[client][phase][image] = response
    # responses = {}
    # responses_practice = {}
    # dict of dicts e.g. responses[client][phase] = [questions]
    # test_results = {}
    # test_results_practice = {}
    condition = None

    def __init__(self, ui=None, condition=None, no_log=False, uid=None):
        import time
        self.clients = {}
        self.ui = ui
        self.session = None
        if uid is None:
            self.uid = time.strftime("%y%m%d.%H%M%S")
        else:
            self.uid = uid
        conditions = ["1", "2", "1r", "2r"]
        if condition not in conditions:
            raise Exception(
                "Invalid condition %s. Should be one of %s" % (condition, conditions))
        self.condition = condition
        self.mode = constants.INIT
        log.startLogging(sys.stdout)
        if not no_log:
            log.addObserver(
                FileLogObserver(open("logs/%s.log" % self.uid, 'w')).emit)
        # self.phase = 0
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

    def stopFactory(self):
        # for lst in self.clients.values():
        #     for client in lst:
        #         client.transport.loseConnection()
        for client in self.clients:
            client.transport.loseConnection()


def get_server_instance(condition, ui=None):
    """
    Returns a default server instance, reading settings 
    from constants.py
    """
    endpoint = TCP4ServerEndpoint(reactor, constants.leap_port)
    factory = LeapP2PServerFactory(ui=ui, condition=condition, no_log=True)
    listener = endpoint.listen(factory)
    factory.endpoint = endpoint
    factory.listener = listener
    return factory


def start_server(qapplication, condition='1', no_ui=False):
    from leaparticulator.p2p.ui.server import LeapP2PServerUI

    factory = None
    assert condition in ['1', '1r', '2', '2r', 'master']
    try:
        if no_ui:
            print "Headless mode..."
            sys.stdout.flush()
            factory = get_server_instance(condition=condition, ui=None)
            if not (constants.TESTING or reactor.running):
                print "Starting reactor..."
                reactor.runReturn()
        else:
            print "Normal GUI mode..."
            sys.stdout.flush()
            ui = LeapP2PServerUI(qapplication)
            factory = get_server_instance(condition=condition, ui=ui)
            ui.setFactory(factory)
            if not (constants.TESTING or reactor.running):
                print "Starting reactor..."
                reactor.runReturn()
            ui.go()
        return factory
    except IndexError, e:
        import traceback
        traceback.print_exc()
        print "ERROR: You should specify a condition (1/2/1r/2r) as a command line argument."
        sys.exit(-1)


if __name__ == '__main__':
    import sys
    try:
        assert len(sys.argv) > 2
        no_ui = sys.argv[-1] == "no_ui"
        if sys.argv[2] == "client":
            assert len(sys.argv) > 3
            try:
                constants.leap_server = sys.argv[4]
            except IndexError:
                pass
            start_client(qapplication, uid=sys.argv[3])
        elif sys.argv[2] == "server":
            start_server(qapplication, condition=sys.argv[1], no_ui=no_ui)

    except AssertionError:
        print "USAGE: LeapP2PServer {condition} {client/server} [client_id] [server_ip (only in client mode)]."
        sys.exit(-1)

    sys.exit(app.exec_())
