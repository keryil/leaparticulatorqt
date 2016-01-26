from PyQt4 import QtGui
from twisted.internet import defer

from test_server_basic import prep, P2PTestCase
from leaparticulator import constants
from leaparticulator.p2p.ui.client import LeapP2PClientUI


class TwoClientsFirstRound(P2PTestCase):

    def tearDown(self):
        self.stopServer()
        for client in self.clients.values():
            client.connection.stopListening()
        self.clients = {}

    def setUp(self):
        from twisted.internet import reactor
        self.reactor = reactor
        prep(self)
        self.startServer()
        self.timeout = 5

        clients = self.startClients(2)

        for client in clients:
            self.clients[client.client_id] = client
            button = client.factory.ui.firstWin.findChildren(
                QtGui.QPushButton, "btnOkay")[0]
            self.click(button)
        # d = defer.Deferred()
        # self.reactor.callLater(.2, lambda: d.callback('setUp'))
        return clients[-1].connection

    def test_createFirstSignal(self):
        self.click(self.factory.ui.mainWin.btnStart)
        d = defer.Deferred()

        def fn():
            ui_speaker, ui_listener = self.getClientsAsUi(0)
            win_speaker = ui_speaker.creationWin
            get_btn = lambda name: win_speaker.findChildren(
                QtGui.QPushButton, name)[0]

            record_btn = get_btn("btnRecord")
            play_btn = get_btn("btnPlay")
            submit_btn = get_btn("btnSubmit")
            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]

            # submit and play start off disabled
            self.assertFalse(play_btn.isEnabled())
            self.assertFalse(submit_btn.isEnabled())

            # record something
            from leaparticulator.data.frame import generateRandomSignal
            self.click(record_btn)
            self.click(record_btn)
            ui_speaker.theremin.last_signal = generateRandomSignal(10)

            # now things should be enabled
            self.assertTrue(play_btn.isEnabled())
            self.assertTrue(submit_btn.isEnabled())

            self.click(submit_btn)
            self.assertTrue(ui_speaker.is_waiting())
            d.callback(("FirstSignal"))
        self.reactor.callLater(.1, fn)

        return d

    def test_FirstImage(self):
        self.click(self.factory.ui.mainWin.btnStart)
        d = defer.Deferred()

        def fn():
            print self.factory.mode
            self.assertEqual(self.factory.mode, constants.SPEAKERS_TURN)
            speaker, listener = self.getClientsAsServerConnections(0)
            speaker_id, listener_id = [c.factory.clients[c] for c in (speaker, listener)]
            print self.clients
            assert isinstance(self.clients, dict)
            ui_speaker, ui_listener = self.getClientsAsUi(0)
            # ui_speaker = self.clients[speaker_id].factory.ui
            # ui_listener = self.clients[listener_id].factory.ui
            self.assertIsInstance(ui_listener, LeapP2PClientUI)

            self.assertEqual(speaker.factory.mode, constants.SPEAKERS_TURN)
            self.assertIsNotNone(speaker)
            self.assertIsNotNone(listener)

            # ui_speaker = speaker.factory.ui
            # ui_listener = listener.factory.ui

            self.assertEqual(
                ui_speaker.get_active_window(), ui_speaker.creationWin)
            self.assertEqual(
                ui_listener.get_active_window(), ui_listener.firstWin)
            self.assertTrue(ui_listener.is_waiting())
            self.assertFalse(ui_speaker.is_waiting())

            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]
            self.assertEqual(self.getRound(0).image.pixmap().toImage(), image.pixmap().toImage())
            d.callback('FirstImage')
        self.reactor.callLater(.1, fn)
        return d

    def test_answerFirstQuestion(self):
        self.click(self.factory.ui.mainWin.btnStart)
        # d_create = defer.Deferred()

        def create():
            # speaker, listener = self.getClientsAsClientData()

            ui_speaker, ui_listener = self.getClientsAsUi(rnd_no=0)

            submit_btn = ui_speaker.creationWin.findChildren(
                QtGui.QPushButton, "btnSubmit")[0]
            record_btn = ui_speaker.creationWin.findChildren(
                QtGui.QPushButton, "btnRecord")[0]

            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]
            # record something
            from leaparticulator.data.frame import generateRandomSignal
            self.click(record_btn)
            self.click(record_btn)
            ui_speaker.theremin.last_signal = generateRandomSignal(2)

            self.click(submit_btn)
            self.reactor.callLater(.5, answer)
        self.reactor.callLater(.2, create)

        d_answer = defer.Deferred()

        def answer():
            # speaker, listener = self.getClientsAsClientData()

            ui_speaker, ui_listener = self.getClientsAsUi(0)
            get_btn = lambda name: ui_listener.testWin.findChildren(
                QtGui.QPushButton, name)[0]

            play_btn = get_btn("btnPlay")
            submit_btn = get_btn("btnSubmit")
            # record_btn = get_btn("btnRecord")
            choices = [get_btn("btnImage%d" % i) for i in range(1, 5)]
            self.click(play_btn)
            self.click(choices[0])
            print "Chosen the answer..."
            def submit():
                self.assertTrue(submit_btn.isEnabled())
                print "Clicking submit button, which is *enabled*"
                self.click(submit_btn)
                d_answer.callback("FirstAnswer")
            self.reactor.callLater(.4, submit)
        # d_create.addCallback(answer)
        # self.reactor.callLater(1, answer)
        # return defer.DeferredList([d_answer, d_create])
        return d_answer
