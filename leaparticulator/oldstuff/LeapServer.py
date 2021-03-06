'''
Created on Feb 18, 2014

@author: kerem
'''
import sys

from PyQt4.QtGui import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print "LeapServer new QApp: %s" % app
else:
    print "LeapServer existing QApp: %s" % app

once = False
# if "twisted.internet.reactor" not in sys.modules:
# from twisted.internet import gtk2reactor
#     gtk2reactor.install()
#     once = True

from twisted.internet import protocol, reactor
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.python import log
from twisted.python.log import FileLogObserver
from twisted.protocols import basic
from leaparticulator import constants
import jsonpickle
import leaparticulator.question as TestQuestion
from leaparticulator.data.meaning import FeaturelessMeaning


class LeapServer(basic.LineReceiver):
    other_end = ""
    delimiter = "\n\n"
    #     MAX_LENGTH = 1024*1024*10
    response = []
    phase = 0
    # image_mask_1 = "./img/meanings/5_*.png"
    # image_mask_2 = "./img/meanings/*_[135].png"
    # image_mask_3 = "./img/meanings/*_[135].png"

    recording = False

    # number of test questions per phase
    n_of_test_questions = constants.n_of_test_questions
    # number of options each test question has
    n_of_options = constants.n_of_options
    # number of total available images per phase
    n_of_meanings = constants.n_of_meanings

    # callbacks
    connectionMadeListeners = set()
    connectionLostListeners = set()

    def __init__(self, factory):  # , image_mask="./img/animals/*.png"):
        self.factory = factory
        from random import shuffle, sample

        files_ = set(range(1, 16))
        files = [None, None, None]
        files[0] = set(sample(files_, 5))
        files[1] = files[0].union(set(sample(files_.difference(files[0]), 5)))
        files[2] = files_

        for i in range(3):
            self.factory.images[i] = [FeaturelessMeaning(n) for n in files[i]]

        shuffle(self.factory.images[0])
        shuffle(self.factory.images[1])
        shuffle(self.factory.images[2])
        print "Images in place!"
        log.msg(self.factory.images)

    def addListenerConnectionMade(self, callback):
        log.msg("Added connectionMade listener.")
        self.connectionMadeListeners.add(callback)

    def addListenerConnectionLost(self, callback):
        log.msg("Added connectionLost listener.")
        self.connectionLostListeners.add(callback)

    def removeListenerConnectionMade(self, callback):
        self.connectionMadeListeners.remove(callback)

    def removeListenerConnectionLost(self, callback):
        self.connectionLostListeners.remove(callback)

    def connectionMade(self):
        self.factory.clients.add(self)
        self.other_end = self.transport.getPeer().host

        # check if we need a new session
        if self.other_end not in self.factory.responses:
            self.factory.responses[self.other_end] = {0: {}, 1: {}, 2: {}}
            self.factory.responses_practice[
                self.other_end] = {0: {}, 1: {}, 2: {}}
        log.msg("Connection with %s is made" % self.other_end)
        for callback in self.connectionMadeListeners:
            callback(self)
        self.start(practice=True)

    def connectionLost(self, reason):
        self.factory.clients.remove(self)
        for callback in self.connectionLostListeners:
            callback(self, reason)
        log.msg("Connection with %s is lost because: %s" %
                (self.other_end, reason))

    def start(self, practice=False):
        self.factory.practice = practice
        if self.phase == 0:
            self.send_all(constants.IMAGE_LIST)
            self.send_all(jsonpickle.encode(self.factory.images))

        self.send_all(constants.START)
        if not practice:
            self.send_all(constants.START_PHASE)
        else:
            self.send_all(constants.START_PRACTICE_PHASE)
        self.factory.mode = constants.LEARN

    def send_all(self, message):
        """
        Send a message to all connected clients
        """
        for c in self.factory.clients:
            c.sendLine(message)

    def sendLine(self, line):
        print "Sending line: %s" % line
        basic.LineReceiver.sendLine(self, str(line))

    def nextPicture(self):
        """
        We need to check if this request is redundant, and if we need
        to switch to a new phase or the testing portion of this phase
        """
        log.msg("nextPicture() called")
        # if we are in the learning phase
        if self.factory.mode == constants.LEARN:
            # print self.factory.image_index,
            # self.factory.responses[self.other_end][self.phase]
            if self.factory.image_index in self.factory.responses[self.other_end][self.phase]:
                if self.factory.responses[self.other_end][self.phase][self.factory.image_index]:
                    # if there are more images to go
                    if self.factory.image_index < len(self.factory.images[self.phase]) - 1:
                        self.factory.image_index += 1
                        log.msg("Redundancy check for next pic passed")
                    # if we ran out of images
                    else:
                        log.msg("End of pictures for this phase.")
                        self.factory.image_index = 0
                        log.msg("End of practice; now for the testing phase")
                        self.factory.mode = constants.TEST
                        self.send_all(constants.TEST)
            else:
                log.msg("No response submitted, sending the same picture")
            if self.factory.mode == constants.LEARN:
                log.msg("Sending next picture: %s" %
                        self.factory.images[self.phase][self.factory.image_index])
                self.send_all(constants.START_NEXT_PIC)
                pic = jsonpickle.encode(
                    self.factory.images[self.phase][self.factory.image_index])
                self.send_all(pic)
                self.send_all(constants.END_NEXT_PIC)

        if self.factory.mode == constants.TEST:
            # if questions are not prepared
            if not self.factory.questions_by_phase[self.phase]:
                # self.factory.questions_by_phase[self.phase] = [TestQuestion(self.factory.responses[self.other_end]) for i in range(self.n_of_test_questions)]
                self.factory.questions_by_phase[self.phase] = TestQuestion.produce_questions(
                    self.factory.responses[self.other_end][self.phase],
                    qty=self.n_of_test_questions[
                        self.phase],
                    n_of_images=self.n_of_options[self.phase])
                print "Generated questions (%d): %s" % (len(self.factory.questions_by_phase[self.phase]),
                                                        self.factory.questions_by_phase[self.phase])
            for q in self.factory.questions_by_phase[self.phase]:
                q.signal = []
                # if this is the first question of the run
                if self.other_end not in self.factory.test_results:
                    self.factory.test_results[
                        self.other_end] = {0: [], 1: [], 2: []}
                    self.factory.test_results_practice[
                        self.other_end] = {0: [], 1: [], 2: []}

                # if this question has already been answered
                if q in self.factory.test_results[self.other_end][self.phase]:
                    continue
                # send the question
                else:
                    self.send_all(constants.START_QUESTION)
                    encoded = jsonpickle.encode(q)
                    # log.msg("Sending question of length %i" % len(encoded))
                    self.send_all(encoded)
                    self.send_all(constants.START_SIGNAL)
                    for s in self.factory.responses[self.other_end][self.phase][q.answer]:
                        self.send_all(jsonpickle.encode(s))
                        q.signal.append(s)
                    self.send_all(constants.END_QUESTION)
                    # log.msg("Number of newlines is %i" % encoded.count("\n"))
                    self.factory.test_results[
                        self.other_end][self.phase].append(q)
                    break
            else:
                if not self.factory.practice:
                    log.msg("End of test, end of phase.")
                    if self.phase < 3:
                        self.factory.image_index = 0
                        self.phase += 1
                        self.send_all(constants.END_OF_PHASE)
                        self.factory.practice = True
                        if self.phase < 3:
                            self.start(practice=True)
                        else:
                            log.msg("End of experiment.")
                            self.send_all(constants.EXIT)
                            import time

                            session_id = self.factory.uid
                            with open("logs/%s.%s.exp.log" % (session_id, self.factory.condition), "w") as f:
                                f.write(jsonpickle.encode(self.factory.images))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.responses))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.test_results))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.responses_practice))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.test_results_practice))
                            return
                else:
                    log.msg("End of practice, start the real phase.")
                    if self.phase < 3:
                        self.factory.test_results_practice[self.other_end][
                            self.phase] = self.factory.test_results[self.other_end][self.phase]
                        self.factory.test_results[
                            self.other_end][self.phase] = []

                        for i in range(3):
                            self.factory.responses_practice[self.other_end][self.phase][
                                i] = self.factory.responses[self.other_end][self.phase][i]
                            self.factory.responses[
                                self.other_end][self.phase][i] = []

                        self.factory.questions_by_phase[self.phase] = []
                        self.factory.practice = False

                        self.factory.image_index = 0
                        # self.phase += 1
                        # self.send_all(Constants.END_OF_PHASE)
                        if self.phase < 3:
                            self.start()
                        else:
                            log.msg("End of experiment.")
                            self.send_all(constants.EXIT)
                            import time
                            from os.path import join

                            session_id = self.factory.uid
                            with open(join("logs", self.factory.log_prefix,
                                           "%s.%s.exp.log" % (session_id, self.factory.condition)), "w") as f:
                                f.write(jsonpickle.encode(self.factory.images))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.responses))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.test_results))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.responses_practice))
                                f.write("\n")
                                f.write(
                                    jsonpickle.encode(self.factory.test_results_practice))
                            return

    def lineReceived(self, line):
        nline = "<{}> {}".format(self.other_end, line)
        # if len(nline) < 300:
        log.msg("Received: %s" % nline)

        #         if line=
        if line == constants.START_REC:
            self.recording = True
            self.response = []
        elif line == constants.END_REC:
            self.recording = False
            self.factory.responses[self.other_end][self.phase][
                self.factory.image_index] = self.response
            # self.nextPicture()
        #             print self.response
        elif line == constants.REQ_NEXT_PIC:
            self.nextPicture()
            return
        elif line == constants.START_RESPONSE:
            assert self.factory.mode == constants.TEST
            self.factory.mode = constants.INCOMING_RESPONSE
        elif line == constants.END_RESPONSE:
            assert self.factory.mode == constants.INCOMING_RESPONSE
            self.factory.mode = constants.TEST
        elif self.factory.mode == constants.INCOMING_RESPONSE:
            q = self.factory.test_results[self.other_end][self.phase][-1]
            assert q.given_answer is None
            answer = jsonpickle.decode(line)
            log.msg("Got answer %s (expected %s)." % (answer, q.answer))
            q.given_answer = answer

        else:
            if self.recording:
                self.response.append(line)


class LeapServerFactory(protocol.Factory):
    numConnections = 0
    protocol = LeapServer
    # Mode is either play or listen
    mode = constants.LEARN
    image_index = 0
    images = [[], [], []]
    questions_by_phase = [[], [], []]
    ui = None
    uid = None
    practice = False

    # dict of dicts e.g. responses[client][phase][image] = response
    responses = {}
    responses_practice = {}
    # dict of dicts e.g. responses[client][phase] = [questions]
    test_results = {}
    test_results_practice = {}
    condition = None

    def __init__(self, ui=None, condition=None, prefix=""):
        import time
        import os
        from os.path import join, isdir

        self.clients = set()
        self.ui = ui
        self.uid = time.strftime("%y%m%d.%H%M%S")
        conditions = ["1", "2", "1r", "2r"]
        if condition not in conditions:
            raise Exception(
                "Invalid condition %s. Should be one of %s" % (condition, conditions))
        self.condition = condition
        if not isdir(join(os.getcwd(), "logs", prefix)):
            os.makedirs(join(os.getcwd(), "logs", prefix))

        log.startLogging(sys.stdout)
        log.addObserver(
            FileLogObserver(open(join("logs", prefix, "%s.log" % self.uid), 'w')).emit)
        log.msg("Condition: %s" % condition)

    def buildProtocol(self, addr):
        server = LeapServer(self)
        LeapServer.MAX_LENGTH = 100000 * 1024
        LeapServer.delimiter = "\n\n"
        server.delimiter = "\n\n"
        server.MAX_LENGTH = 100000 * 1024
        if self.ui:
            server.addListenerConnectionLost(self.ui.connectionLost)
            server.addListenerConnectionMade(self.ui.connectionMade)
        return server


class LeapClient(basic.LineReceiver):
    delimiter = "\n\n"

    def __init__(self, factory):
        self.factory = factory

    def connectionMade(self):
        self.factory.resetDelay()
        log.msg("Connection made to %s" % self.transport.getPeer())

    def connectionLost(self, reason):
        log.msg("Connection lost")

    def lineReceived(self, data):
        log.msg("Received: %s" % data)
        if data == constants.START:
            self.factory.mode = constants.LEARN
            return
        elif data == constants.EXIT:
            self.factory.ui.exit()
        elif data == constants.IMAGE_LIST:
            self.factory.mode = constants.IMAGE_LIST
            return
        elif data == constants.START_NEXT_PIC:
            self.factory.prev_mode = self.factory.mode
            self.factory.mode = constants.RECEIVE_PIC
            return
        elif data == constants.END_NEXT_PIC:
            self.factory.mode = self.factory.prev_mode
            return
        elif data == constants.TEST:
            self.factory.mode = constants.TEST
            self.factory.ui.go_test()
        elif data == constants.START_PHASE:
            self.factory.ui.next_phase()
        elif data == constants.START_PRACTICE_PHASE:
            self.factory.ui.next_phase(True)
        elif data == constants.START_QUESTION:
            assert self.factory.mode == constants.TEST
            self.factory.mode = constants.START_QUESTION
            return
        elif data == constants.END_QUESTION:
            self.factory.mode = constants.TEST
            self.factory.ui.on_new_test_question(self.factory.ui.question)
        elif data == constants.START_SIGNAL:
            self.factory.mode = constants.INCOMING_RESPONSE
            return

        if self.factory.mode == constants.RECEIVE_PIC:
            log.msg("New picture received: %s" % data)
            self.factory.current_image = data
            self.factory.ui.on_new_picture(data)
        elif self.factory.mode == constants.IMAGE_LIST:
            log.msg("Image list received: %s" % data)
            self.factory.ui.images = jsonpickle.decode(data)
        elif self.factory.mode == constants.START_QUESTION:
            log.msg("New question received: %s" % jsonpickle.decode(data))
            self.factory.mode = constants.INCOMING_RESPONSE
            self.factory.ui.question = jsonpickle.decode(data)
        elif self.factory.mode == constants.INCOMING_RESPONSE:
            self.factory.ui.question.signal.append(jsonpickle.decode(data))


class LeapClientFactory(protocol.ReconnectingClientFactory):
    protocol = LeapClient
    mode = None
    current_image = None
    # this is the leap hearer
    listener = None
    ui = None
    images = None

    def __init__(self, leap_listener, ui):
        if not ui:
            print "Whaddahell, no UI??"
        self.listener = leap_listener
        self.ui = ui

    def buildProtocol(self, addr):
        c = LeapClient(self)
        self.listener.protocol = c
        log.msg("New LeapClient initialized")
        return c


def get_server_instance():
    """
    Returns a default server instance, reading settings 
    from constants.py
    """
    endpoint = TCP4ServerEndpoint(reactor, constants.leap_port)
    endpoint.listen(LeapServerFactory())
    return endpoint


if __name__ == '__main__':
    import sys

    endpoint = TCP4ServerEndpoint(reactor, constants.leap_port)
    if len(sys.argv) < 2:
        print "ERROR: You should specify a condition (1/2/1r/2r) as a command line argument."
        sys.exit(-1)
    else:
        prefix = ""
        if len(sys.argv) > 2:
            prefix = sys.argv[2]
        from os.path import join
        from os import getcwd

        print "**** IMPORTANT: Log folder is %s ****" % join(getcwd(), "logs", prefix)
        endpoint.listen(
            LeapServerFactory(condition=sys.argv[1], prefix=prefix))

    reactor.run()
