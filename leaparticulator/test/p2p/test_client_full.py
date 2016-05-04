from PyQt4 import QtGui
from twisted.internet import defer

from leaparticulator import constants
from leaparticulator.p2p.ui.client import LeapP2PClientUI
from test_server_basic import P2PTestCase


class TwoClientsFirstRound(P2PTestCase):

    def test_createFirstSignal(self):
        # self.click(self.factory.ui.mainWin.btnStart)
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
            ui_speaker.theremin.last_signal = generateRandomSignal(10)
            self.click(record_btn)

            # now things should be enabled
            self.assertTrue(play_btn.isEnabled())
            self.assertTrue(submit_btn.isEnabled())

            self.click(submit_btn)
            self.assertTrue(ui_speaker.is_waiting())
            return d.callback(("FirstSignal"))

        self.reactor.callLater(.2, fn)
        return d

    def test_FirstImage(self):
        # self.click(self.server_factory.ui.mainWin.btnStart)
        d = defer.Deferred()

        def fn():
            print self.server_factory.mode
            self.assertEqual(self.server_factory.mode, constants.SPEAKERS_TURN)
            speaker, listener = self.getClientsAsServerConnections(0)
            speaker_id, listener_id = [c.factory.clients[c] for c in (speaker, listener)]
            print "FirstImage clients:", self.clients
            assert isinstance(self.clients, dict)
            ui_speaker, ui_listener = self.getClientsAsUi(0)
            # ui_speaker = self.clients[speaker_id].factory.ui
            # ui_listener = self.clients[listener_id].factory.ui
            self.assertIsInstance(ui_listener, LeapP2PClientUI)

            self.assertEqual(speaker.factory.mode, constants.SPEAKERS_TURN)
            self.assertIsNotNone(speaker)
            self.assertIsNotNone(listener)

            # ui_speaker = speaker.factory.ui
            # ui_listener = hearer.factory.ui

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

        self.reactor.callLater(.2, fn)
        return d

    def test_answerFirstQuestion(self):
        d = defer.Deferred()

        def check(*args):
            assert len(self.getRounds()) == 1
            round = self.getLastRound()
            self.assertEqual(round.image, round.guess)
            self.assertTrue(round.success)
            for factory in self.factories:
                self.assertTrue(factory.mode == constants.FEEDBACK)
            d.callback("Done")

        self.do_one_round(callback=check)
        return d

    def test_answerTwoQuestions(self):
        d = defer.Deferred()

        def do_test(*args):
            assert len(self.getRounds()) == 2
            round = self.getLastRound()
            self.assertEqual(round.image, round.guess)
            self.assertTrue(round.success)
            for factory in self.factories:
                self.assertTrue(factory.mode == constants.FEEDBACK)
            d.callback("Done")

        def do_second(*args):
            self.do_one_round(callback=do_test)

        self.do_one_round(callback=do_second)
        return d


class TwoClientsTillEnd(P2PTestCase):
    def __init__(self, *args, **kwargs):
        super(TwoClientsTillEnd, self).__init__(*args, **kwargs)
        self.max_images = 15
        self.timeout = self.max_images * 8

    def test_endByExhaustion(self):
        print "Timeout set to: {}".format(self.timeout)
        d = defer.Deferred()

        def do_tests(*args):
            for c in self.getClientsAsUi():
                assert c.get_active_window() == c.finalScreen
            d.callback("Done")

        def exhaust(round=-1):
            print "-------------------EXHAUST CALLED-------------------"
            if self.server_factory.end_experiment:
                print "Exhaustion complete at round #{}.".format(round)
                self.reactor.callLater(.2, do_tests)
            else:
                self.do_one_round().addCallback(lambda *x: exhaust(round + 1))

        exhaust()
        return d
