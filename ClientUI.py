'''
Created on Feb 24, 2014

@author: kerem
'''

from collections import deque

import pygtk

from LeapTheremin import gimmeSomeTheremin, ThereminPlayback
from leaparticulator import constants

pygtk.require("2.0")

import gtk

from twisted.python import log
import jsonpickle
from AbstractClientUI import AbstractClientUI


class ClientUI(AbstractClientUI):
    shapeArea = None
    learningWindow = None
    volumeControls = []
    answerButtons = []
    size = 1.1
    w, h = 300, 300
    theremin = None
    reactor = None
    controller = None
    recording = False
    playback_player = None
    phase = 0
    last_signal = []
    n_of_test_questions = 4
    activeWindow = None
    activeHandler = None
    shownPretest = False
    practicePhase = False
    default_volume = 0.5
    default_pitch = 440.

    # 1,1r,2,2r
    condition = None

    def __init__(self, condition):
        condition = str(condition)
        conditions = ["1", "2", "1r", "2r"]
        if condition not in conditions:
            raise Exception("Invalid condition %s. Should be one of %s" % (condition, conditions))
        self.condition = condition

        # Set the Glade file
        self.gladefile = "ClientWindow.glade"
        self.builder = builder = gtk.Builder()
        builder.add_from_file(self.gladefile)
        # builder.connect_signals(self)


        #         print self.theremin, self.reactor, self.controller, self.connection
        #         while not self.theremin:
        #             pass
        self.backgroundWindow = builder.get_object("backgroundWindow")
        self.backgroundWindow.maximize()
        self.backgroundWindow.set_keep_above(True)
        self.backgroundWindow.show_all()

        self.learningWindow = builder.get_object("learningWindow")
        self.testingWindow = builder.get_object("testingWindow")
        self.firstWindow = builder.get_object("firstWindow")
        self.prePhaseWindow = builder.get_object("prePhaseWindow")
        self.preTestWindow = builder.get_object("preTestWindow")
        self.finalWindow = builder.get_object("finalWindow")
        self.phaseOne = builder.get_object("phaseOne")
        self.phaseTwo = builder.get_object("phaseTwo")
        self.phaseThree = builder.get_object("phaseThree")
        self.btnSubmitTest = builder.get_object("btnSubmitTest")
        self.btnReplayTest = builder.get_object("btnReplayTest")
        self.defaultButtonStyle = self.btnSubmitTest.get_style().copy()

        self.instructionsPhaseOne = builder.get_object("instructionTextPhaseOne")
        self.instructionsPhaseTwo = builder.get_object("instructionTextPhaseTwo")
        self.instructionsPhaseThree = builder.get_object("instructionTextPhaseThree")

        builder.get_object("btnOkayPhaseOne").connect("clicked", self.go_phase_one)
        builder.get_object("btnOkayPhaseTwo").connect("clicked", self.go_phase_two)
        builder.get_object("btnOkayPhaseThree").connect("clicked", self.go_phase_three)
        builder.get_object("btnSubmit").connect("clicked", self.on_btnSubmit_clicked)
        builder.get_object("btnReplay").connect("clicked", self.on_btnReplay_clicked)
        builder.get_object("btnRecord").connect("clicked", self.on_btnRecord_clicked)
        builder.get_object("btnOkayPrePhase").connect("clicked", self.on_btnOkayPrePhase_clicked)
        builder.get_object("btnOkayPreTest").connect("clicked", self.go_test)
        builder.get_object("btnQuit").connect("clicked", gtk.main_quit)
        builder.get_object("firstButton").connect("clicked", self.go_pre_phase)

        for i in range(1, 10):
            c = builder.get_object("volumeControl%i" % i)
            if c:
                self.volumeControls.append(c)
                self.change_widget_background(c, "orange")
                c.connect("value-changed", self.on_volume_change)
                c.connect("popup", self.unfocus_main)
                c.connect("popdown", self.focus_main)
            else:
                break

        self.theremin, self.reactor, self.controller, self.connection = gimmeSomeTheremin(n_of_notes=1,
                                                                                          default_volume=self.default_volume,
                                                                                          ui=self)
        self.playback_player = ThereminPlayback()
        self.set_textview_font()

    def set_textview_font(self, font_name=None):
        for widget in self.builder.get_objects():
            if isinstance(widget, gtk.TextView):
                self.set_font(widget, font_name)

    def set_font(self, widget, font_name=None):
        if not font_name:
            font = widget.get_style().font_desc.copy()
            print "Current font is %s" % font.to_string()
            font.set_size(12 * 1024)
            print "Current font is %s" % font.to_string()
            widget.modify_font(font)
        else:
            font = pango.FontDescription(font_name)
            widget.modify_font(font)

    def on_bg_focus(self, window, event):
        self.activeWindow.present()
        self.activeWindow.queue_draw()

    def unfocus_main(self, button):
        self.dummy = self.activeWindow
        self.activeWindow.set_keep_above(False)
        self.activeWindow = button.get_popup()
        self.activeWindow.set_keep_above(True)

    def focus_main(self, button):
        self.activeWindow = self.dummy
        self.activeWindow.set_keep_above(True)


    def show_window(self, window):
        """
        Shows the window and sets up necessary settings such 
        as stay-on-top.
        """

        def stay_on_top(window, event):
            window.present()
            window.queue_draw()

        if self.activeWindow:
            self.activeWindow.disconnect(self.activeHandler)
            self.activeWindow.set_keep_above(False)
            self.activeWindow.hide()
        self.activeWindow = window
        self.activeHandler = window.connect("focus-out-event", stay_on_top)
        self.activeWindow.set_keep_above(True)
        window.connect("destroy", self.on_mainWindow_destroy)
        window.set_modal = True
        window.show_all()


    def on_volume_change(self, widget, volume):
        self.theremin.player.volume_coefficient = max(0.01, volume)
        self.playback_player.player.volume_coefficient = max(0.01, volume)
        for control in self.volumeControls:
            control.value = volume

    def reset_last_signal(self):
        self.last_signal = deque()

    def extend_last_signal(self, frame_as_json):
        self.last_signal.append(frame_as_json)

    def request_new_picture(self):
        self.reset_last_signal()
        if self.theremin.protocol.factory.ui == None:
            self.theremin.protocol.factory.ui = self
        self.send(constants.REQ_NEXT_PIC)

    def on_new_test_question(self, question):
        grid = self.builder.get_object("testGrid")
        grid.resize(question.n_of_images / 2, 2)
        for o in self.answerButtons:
            o.destroy()
        self.question = question
        self.question_images = []
        log.msg("Current images at phase %i are %s" % (self.phase, self.images[self.phase - 1]))
        for i, pic in enumerate(question.pics):
            log.msg("Added new response image at %i-%i,%i-%i: %s" % (i % 2, i % 2 + 1,
                                                                     i / 2, i / 2 + 1, pic))
            img = gtk.Image()

            img.set_from_file(self.images[self.phase - 1][pic])
            imgButton = gtk.Button()
            self.change_widget_background(imgButton, "white")
            imgButton.set_name("answer%s" % pic)
            log.msg("Added answer button %s" % imgButton.get_name())
            imgButton.set_image(img)
            grid.attach(imgButton,
                        i % 2, i % 2 + 1,
                        i / 2, i / 2 + 1)
            img.show()
            imgButton.show()
            imgButton.connect("clicked", self.on_click_test_picture, img)
            self.answerButtons.append(imgButton)
            # print img.path()
        self.builder.get_object("shapeFrameTest").set_label("Choose the shape that this signal denotes.")

        # To fix #24
        if self.activeWindow == self.testingWindow:
            self.playback_player.start(self.question.signal)
        self.btnReplayTest.set_sensitive(True)
        self.testingWindow.queue_draw()

    def on_click_test_picture(self, widget, img):
        self.question.given_answer = int(widget.get_name()[6:])
        self.btnSubmitTest.set_sensitive(True)
        for p in self.answerButtons:
            self.change_widget_background(p, "grey")
        self.change_widget_background(widget, "blue")
        print "Answer is %s" % self.question.given_answer

    def on_btnOkayPrePhase_clicked(self, widget):
        log.msg("Prephase okay button clicked.")
        self.go_phase()

    def on_new_picture(self, pic):
        log.msg("Displaying new picture %s" % pic)
        if self.phase == 1:
            # if self.theremin.protocol.factory.mode != Constants.TEST:
            self.builder.get_object("imageLearn").set_from_file(pic)
            self.builder.get_object("btnRecord").set_label("Record")
            self.learningWindow.queue_draw()
        elif self.phase == 2:
            self.builder.get_object("imageLearn").set_from_file(pic)
            self.builder.get_object("btnRecord").set_label("Record")
            self.learningWindow.queue_draw()
        elif self.phase == 3:
            self.builder.get_object("imageLearn").set_from_file(pic)
            self.builder.get_object("btnRecord").set_label("Record")
            self.learningWindow.queue_draw()
        else:
            raise Exception("Invalid phase: %s" % self.phase)

        #

    def next_phase(self, practice=False):
        """
        This method either steps the client through the first 
        window or takes them to the next phase's pre-phase 
        screen.
        """
        self.practicePhase = practice
        if not self.practicePhase:
            log.msg("New real phase: %i" % self.phase)
            self.go_phase()
        else:
            self.phase += 1
            if self.phase > 3:
                self.exit()
                return
            log.msg("New practice phase: %i" % self.phase)
            if self.phase == 1:
                self.go_first_window()
            else:
                self.go_pre_phase()


    def go_phase(self):
        """
        Initiates the info window of the current phase.
        """
        self.prePhaseWindow.hide()
        if self.phase == 1:
            self.go_phase_one_info()
        elif self.phase == 2:
            log.msg("Calling go_phase_two_info()")
            self.go_phase_two_info()
        elif self.phase == 3:
            self.go_phase_three_info()

    def go_test(self, dummy=None):
        """
        Initiates the test portion of the current phase.
        """
        log.msg("Starting test for phase %i" % self.phase)
        self.test_images = []
        if self.shownPretest:
            if self.phase == 1:
                self.go_phase_one_test()
            elif self.phase == 2:
                self.go_phase_two_test()
            elif self.phase == 3:
                self.go_phase_three_test()

            self.builder.get_object("btnReplayTest").connect("clicked", self.on_click_test_replay)
            self.builder.get_object("btnSubmitTest").connect("clicked", self.on_click_test_submit)
            self.shownPretest = False
            self.playback_player.start(self.question.signal)
        else:
            self.preTestWindow.set_title("Testing Phase %i" % self.phase)
            self.show_window(self.preTestWindow)
            self.shownPretest = True

    def exit(self):
        """
        Shows the final screen.
        """
        self.show_window(self.finalWindow)

    def on_click_test_replay(self, widget):
        self.playback_player.start(self.question.signal)

    def change_widget_background(self, widget, color):
        """
        Changes the background of a GTK widget to that 
        denoted by the parameter color, which is a string.
        """
        o = widget
        map = o.get_colormap()
        color = map.alloc_color(color)
        # copy the current style and replace the background
        style = o.get_style().copy()
        style.bg[gtk.STATE_NORMAL] = color
        style.bg[gtk.STATE_PRELIGHT] = color
        style.bg[gtk.STATE_ACTIVE] = color
        style.bg[gtk.STATE_SELECTED] = color
        style.bg[gtk.STATE_INSENSITIVE] = color
        #set the button's style to the one you created
        o.set_style(style)
        o.queue_draw()

    def change_widget_foreground(self, widget, color):
        """
        Changes the background of a GTK widget to that 
        denoted by the parameter color, which is a string.
        """
        o = widget
        map = o.get_colormap()
        color = map.alloc_color(color)
        # copy the current style and replace the background
        style = o.get_style().copy()
        style.fg[gtk.STATE_NORMAL] = color
        style.fg[gtk.STATE_PRELIGHT] = color
        style.fg[gtk.STATE_ACTIVE] = color
        style.fg[gtk.STATE_SELECTED] = color
        style.fg[gtk.STATE_INSENSITIVE] = color
        #set the button's style to the one you created
        o.set_style(style)
        o.queue_draw()

    def on_click_test_submit(self, widget):
        if not self.btnSubmitTest.get_sensitive():
            log.msg("Why did you click an object right then?")
            return
        self.send(constants.START_RESPONSE)
        self.send(jsonpickle.encode(self.question.given_answer))
        self.send(constants.END_RESPONSE)

        # change color of buttons for feedback
        correct_answer = self.question.answer
        for o in self.answerButtons:
            o.set_sensitive(False)
            if o.get_name() == "answer%s" % correct_answer:
                self.change_widget_background(o, "green")
                # if correct_answer == self.question.given_answer:
                # break
            elif o.get_name() == "answer%s" % self.question.given_answer:
                self.change_widget_background(o, "red")

        self.btnSubmitTest.set_sensitive(False)
        self.btnReplayTest.set_sensitive(False)
        # self.learningWindow.queue_draw()
        self.reactor.callLater(2, self.send, constants.REQ_NEXT_PIC)

    ## START
    # First screen
    def go_first_window(self):
        self.theremin.player.mute()
        self.show_window(self.firstWindow)
        print self.builder.get_object("firstText").get_style().font_desc

    # Prephase screen for all phases
    def go_pre_phase(self, dummy=None):
        self.prePhaseWindow.set_title("Leap Articulator - Phase %i" % self.phase)

        buf = self.phase

        # if this is the second condition, swap phase3 and phase2
        if self.condition[0] == '2':
            if buf == 3:
                buf = 2
            elif buf == 2:
                buf = 3

        self.theremin.player.default_pitch = self.playback_player.player.default_pitch = None
        self.theremin.player.default_volume = self.playback_player.player.default_volume = None
        # if this is a reversed condition, freeze frequency, 
        # release volume, do the other way around otherwise
        if self.condition[-1] == 'r':
            if (self.condition[0] == '1' and self.phase == 2) or \
                    (self.condition[0] == '2' and self.phase == 3) or \
                            self.phase == 1:
                self.theremin.player.default_pitch = self.playback_player.player.default_pitch = self.default_pitch
            self.theremin.player.default_volume = self.playback_player.player.default_volume = None
        else:
            self.theremin.player.default_pitch = self.playback_player.player.default_pitch = None
            if (self.condition[0] == '1' and self.phase == 2) or \
                    (self.condition[0] == '2' and self.phase == 3) or \
                            self.phase == 1:
                self.theremin.player.default_volume = self.playback_player.player.default_volume = self.default_volume

        buf = "instructionsBuffer%i_1" % buf
        if self.condition[-1] == 'r':
            buf = buf + "_reversed"
        log.msg("Loading pre-phase text buffer \"%s\"" % buf)
        buf = self.builder.get_object(buf)
        self.builder.get_object("instructionTextPrePhase").set_buffer(buf)


        # if self.phase == 2:
        # if self.condition[0] == 1:
        #     self.theremin.player.default_volume = self.playback_player.player.default_volume = None
        # elif self.phase == 3:
        #     self.theremin.player.default_volume = self.playback_player.player.default_volume = self.default_volume

        self.show_window(self.prePhaseWindow)
        self.theremin.player.unmute()

    # Phase one
    def go_phase_one_info(self):
        log.msg("Phase on is a go (practice: %s)." % self.practicePhase)
        inst = None
        if self.practicePhase:
            inst = self.builder.get_object("practiceBuffer%i" % self.phase)
        else:
            inst = self.builder.get_object("instructionsBuffer%i" % self.phase)
        self.instructionsPhaseOne.set_buffer(inst)

        self.show_window(self.phaseOne)
        self.theremin.player.unmute()

    def go_phase_one(self, widget):
        self.request_new_picture()
        self.show_window(self.learningWindow)
        self.theremin.player.mute()
        print "Phase one initiated"

    def go_phase_one_test(self):
        self.show_window(self.testingWindow)
        self.theremin.player.mute()
        print "Phase one test initiated"

    # Phase one
    ## END

    ## START
    # Phase two
    def go_phase_two_info(self):
        # self.theremin.player.default_volume = self.playback_player.player.default_volume = None
        self.theremin.player.unmute()
        buf = self.phase
        if self.condition[0] == '2':
            buf += 1
        inst = None
        if self.practicePhase:
            inst = self.builder.get_object("practiceBuffer%i" % buf)
        else:
            inst = self.builder.get_object("instructionsBuffer%i" % buf)

        if self.condition[0] == '2':
            self.instructionsPhaseThree.set_buffer(inst)
            self.phaseThree.set_title("LeapArticulator - Phase 2")
            self.show_window(self.phaseThree)
        elif self.condition[0] == '1':
            self.instructionsPhaseTwo.set_buffer(inst)
            self.phaseTwo.set_title("LeapArticulator - Phase 2")
            self.show_window(self.phaseTwo)
        log.msg("Phase two info initiated.")

    def go_phase_two(self, widget):
        self.request_new_picture()
        self.learningWindow.set_title("LeapArticulator - Phase 2")
        self.show_window(self.learningWindow)
        self.theremin.player.mute()
        self.request_new_picture()
        print "Phase two initiated"

    def go_phase_two_test(self):
        self.show_window(self.testingWindow)
        self.theremin.player.mute()
        print "Phase two test initiated"

    # Phase two
    ## END

    ## START
    # Phase three
    def go_phase_three_info(self):
        # self.show_window(self.phaseThree)
        self.theremin.player.unmute()
        inst = None
        buf = self.phase
        if self.condition[0] == '2':
            buf = 2
        if self.practicePhase:
            inst = self.builder.get_object("practiceBuffer%i" % buf)
        else:
            inst = self.builder.get_object("instructionsBuffer%i" % buf)

        if self.condition[0] == '2':
            self.instructionsPhaseTwo.set_buffer(inst)
            self.phaseTwo.set_title("LeapArticulator - Phase 3")
            self.show_window(self.phaseTwo)
        elif self.condition[0] == '1':
            self.instructionsPhaseThree.set_buffer(inst)
            self.phaseThree.set_title("LeapArticulator - Phase 3")
            self.show_window(self.phaseThree)

    def go_phase_three(self, widget):
        self.phaseTwo.hide()
        self.learningWindow.set_title("LeapArticulator - Phase 3")
        self.show_window(self.learningWindow)
        self.theremin.player.mute()
        self.request_new_picture()
        print "Phase three initiated"

    def go_phase_three_test(self):
        self.show_window(self.testingWindow)
        self.theremin.player.mute()
        print "Phase three test initiated"

    # Phase three
    ## END

    def on_btnSubmit_clicked(self, widget):
        print "Submit"
        self.request_new_picture()

    def on_btnReplay_clicked(self, widget):
        print "Replay"
        self.playback_player.start(self.last_signal)

    def on_btnRecord_clicked(self, widget):
        if not self.recording:
            self.builder.get_object("btnSubmit").set_sensitive(False)
            self.reset_last_signal()
            widget.set_label("Stop")
            self.send(constants.START_REC)
            self.recording = True
            self.theremin.player.unmute()
            print "Record"
        else:
            self.builder.get_object("btnSubmit").set_sensitive(True)
            widget.set_label("Re-record")
            self.send(constants.END_REC)
            self.recording = False
            self.theremin.player.mute()

    def send(self, msg):
        """
        Sends the message to the server. Convenience method.
        """
        self.theremin.protocol.sendLine(msg)

    def expose_event(self, widget, event):
        print "expose"
        x, y, width, height = event.area
        self.drawRectangle(self.size)
        return gtk.FALSE

    def configure_event(self, widget, event):
        print "configure"
        widget = self.shapeArea
        self.drawRectangle(size=self.size)
        return gtk.TRUE


    def on_mainWindow_destroy(self, window):
        gtk.main_quit()


if __name__ == '__main__':
    # conditions: 1,1r,2,2r
    import sys

    try:
        ui = ClientUI(condition=sys.argv[1])
    except IndexError:
        print "ERROR: You should specify a condition (1/2/1r/2r)"
    from platform import system

    # fixes issue #2
    if system() == "Windows":
        ui.reactor.run()
    else:
        gtk.main()
