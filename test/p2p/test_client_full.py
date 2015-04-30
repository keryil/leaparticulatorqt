from PyQt4 import QtGui
from twisted.internet import defer

from test_server_basic import prep, P2PTestCase
from leaparticulator import constants


class TwoClientsFirstRound(P2PTestCase):

    def tearDown(self):
        self.stopServer()
        self.clients = []

    def setUp(self):
        from twisted.internet import reactor
        self.reactor = reactor
        prep(self)
        self.startServer()
        self.timeout = 3

        self.clients = self.startClients(2)
        for client in self.clients:
            button = client.factory.ui.firstWin.findChildren(
                QtGui.QPushButton, "btnOkay")[0]
            self.click(button)
        d = defer.Deferred()
        self.reactor.callLater(.2, lambda: d.callback('setUp'))
        return d

    def test_createFirstSignal(self):
        self.click(self.factory.ui.mainWin.btnStart)
        d = defer.Deferred()

        def fn():
            speaker, listener = self.getClients()
            ui_speaker = speaker.factory.ui
            ui_listener = listener.factory.ui
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
            from leaparticulator.leap.frame import generateRandomSignal
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
            speaker, listener = self.getClients()

            ui_speaker = speaker.factory.ui
            ui_listener = listener.factory.ui

            self.assertIsNotNone(speaker)
            self.assertEqual(speaker.factory.mode, constants.SPEAKER)
            self.assertIsNotNone(listener)
            self.assertEqual(listener.factory.mode, constants.LISTENER)

            self.assertEqual(
                ui_speaker.get_active_window(), ui_speaker.creationWin)
            self.assertEqual(
                ui_listener.get_active_window(), ui_listener.firstWin)
            self.assertTrue(ui_listener.is_waiting())
            self.assertFalse(ui_speaker.is_waiting())

            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]
            self.assertEqual(
                speaker.factory.current_speaker_image.pixmap().toImage(), image.pixmap().toImage())
            d.callback('FirstImage')
        self.reactor.callLater(.1, fn)
        return d

    def test_answerFirstQuestion(self):
        self.click(self.factory.ui.mainWin.btnStart)
        d_create = defer.Deferred()

        def create():
            speaker, listener = self.getClients()
            ui_speaker = speaker.factory.ui
            ui_listener = listener.factory.ui

            submit_btn = ui_speaker.creationWin.findChildren(
                QtGui.QPushButton, "btnSubmit")[0]
            record_btn = ui_speaker.creationWin.findChildren(
                QtGui.QPushButton, "btnRecord")[0]

            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]
            # record something
            from leaparticulator.leap.frame import generateRandomSignal
            self.click(record_btn)
            self.click(record_btn)
            ui_speaker.theremin.last_signal = generateRandomSignal(10)

            self.click(submit_btn)
            d_create.callback(("FirstAnswer"))
        self.reactor.callLater(.1, create)

        d_answer = defer.Deferred()

        def answer():
            speaker, listener = self.getClients()

            ui_speaker = speaker.factory.ui
            ui_listener = listener.factory.ui
            get_btn = lambda name: ui_listener.testWin.findChildren(
                QtGui.QPushButton, name)[0]

            play_btn = get_btn("btnPlay")
            submit_btn = get_btn("btnSubmit")
            choices = [get_btn("btnImage%d" % i) for i in range(1, 5)]

            image = ui_speaker.creationWin.findChildren(
                QtGui.QLabel, "lblImage")[0]

            # record something
            from leaparticulator.leap.frame import generateRandomSignal
            self.click(record_btn)
            self.click(record_btn)
            ui_speaker.theremin.last_signal = generateRandomSignal(10)

            self.click(submit_btn)
            d_answer.callback("FirstAnswer")
        self.reactor.callLater(.4, answer)
        return defer.DeferredList([d_answer, d_create])
