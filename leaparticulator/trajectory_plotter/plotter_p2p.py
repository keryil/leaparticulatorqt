__author__ = 'kerem'
import sip
import sys, os
sys.path.append(os.getcwd())

try:
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QString', 2)
    sip.setapi('QtextStream', 2)
    sip.setapi('Qtime', 2)
    sip.setapi('QUrl', 2)
    sip.setapi('QVariant', 2)
except ValueError, e:
    raise RuntimeError('Could not set API version (%s): did you import PyQt4 directly?' % e)

from PyQt4 import QtGui

app = QtGui.QApplication(sys.argv)

from leaparticulator import constants

constants.install_reactor()

from PyQt4 import QtGui

from leaparticulator.oldstuff.QtUtils import loadUiWidget

from leaparticulator.data.functions import fromFile
import numpy as np
import imageio as im

class PlotterWindow(QtGui.QMainWindow):
    def __init__(self):
        super(PlotterWindow, self).__init__()
        # self.filename = "/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/leaparticulator/test/test_data/123R0126514.1r.exp.log"
        self.filename = None
        # self.playback = ThereminPlayback(record=False)
        loadUiWidget('Traj2GIF.ui', widget=self)
        self.setWindowTitle("Plotter")

        # self.setLogFile(self.filename)
        # self.score = self.data['1']['./img/meanings/1_1.png']
        import os
        self.txtOutputPath.setText(os.path.abspath(os.path.expanduser("~" + os.sep + "Desktop")))

        self.actionPlay.triggered.connect(self.preview)
        self.actionRecord.triggered.connect(self.produce_selected_gifs)
        self.actionChange_Path.triggered.connect(self.setOutputPath)

        # fix the terribly slow open file dialog under
        # opensuse
        import platform
        options = QtGui.QFileDialog.Options(0)
        if platform.system() == "Linux":
            options = QtGui.QFileDialog.DontUseNativeDialog
        self.actionOpenLog.triggered.connect(
            lambda: self.setLogFile(str(QtGui.QFileDialog.getOpenFileName(options=options))))

        self.score = None
        self.lstSignals.currentItemChanged.connect(self.selectScore)

        # connect the playback rate spinner
        self.spinPlayback.valueChanged.connect(self.setRate)
        # self.spinPlayback.setValue(int(1. / constants.THEREMIN_RATE))

        self.x_offset = 200
        self.y_max = 600
        self.x_max = 600
        self.fps = 30
        self.n_trace = 15
        self.trace_width = 2
        self.hand_width = 10
        self.hand_height = 10

        def set_ntrace(n):
            self.n_trace = n
            print "n_trace set to", n
        def set_trace_width(n):
            self.trace_width = n
            print "Trace width set to", n
        self.spinTrace.valueChanged.connect(set_ntrace)
        self.spinWidth.valueChanged.connect(set_trace_width)

        print "Done!"

    def round(self, flt):
        return int(np.round(flt))

    def updateRateSpin(self):
        self.spinPlayback.setValue(constants.THEREMIN_RATE)

    def setRate(self, rate):
        print "FPS set to", rate
        self.fps = rate

    def selectScore(self, current, previous):
        if current:
            self.setScore(current.signal)

    def setScore(self, score):
        self.score = score

    def setOutputPath(self):
        print self.txtOutputPath.text()
        import platform
        options = QtGui.QFileDialog.Options(0)
        if platform.system() == "Linux":
            options = QtGui.QFileDialog.DontUseNativeDialog
        text = str(QtGui.QFileDialog.getExistingDirectory(options=options))
        self.txtOutputPath.setText(text)

    def setLogFile(self, fname):
        self.data, self.meanings = fromFile(fname)
        self.filename = fname
        self.populateWithSignals()
        self.btnPlay.setEnabled(True)
        self.btnRecord.setEnabled(True)

    def populateWithSignals(self):
        self.items = []
        self.lstSignals.clear()
        for speaker in self.data:
            for phase in self.data[speaker]:
                for meaning, signal in self.data[speaker][phase].items():
                    if signal is not None and signal != []:
                        item = QtGui.QListWidgetItem(
                            "Speaker %s, Phase %s, Meaning %s" % (speaker.split("@")[0], phase, meaning))
                        item.signal = signal
                        item.meaning = meaning
                        item.phase = phase
                        item.speaker = speaker
                        self.items.append(item)
                        self.lstSignals.addItem(item)

    def preview(self):
        print "Preview!"
        from tempfile import NamedTemporaryFile
        f = "temp_file.gif"
        self.render_single_gif(traj=self.score, temp_file=f)
        preview = QtGui.QMovie(f)
        preview.setCacheMode(QtGui.QMovie.CacheAll)

        lblPreview = self.findChild(QtGui.QLabel, "lblPreview")
        lblPreview.setMovie(preview)
        # preview.setScaledSize(lblPreview.sizeHint())
        preview.start()

    def new_image(self, fill=.5, x=0, y=0):
        """
        Returns a new numpy array of given size, filled with given value.
        If either x or y is zero, they are replaced by self.y_max and
        self.x_max.
        :param fill:
        :param x:
        :param y:
        :return:
        """
        # if x + y < 1:
        #     raise Exception("Cannot produce image of size %dx%d" % (x, y))
        if x == 0:
            x = self.x_max
        if y == 0:
            y = self.y_max

        data = np.empty((x, y))
        data.fill(fill)
        return data

    def transform(self, x, y):
        """
        Transforms given Leap coordinates to GIF coordinates.
        :param x:
        :param y:
        :return:
        """
        x += self.x_offset
        if y < 0:
            y = 0
        y = self.y_max - y
        try:
            assert x >= 0
            assert x <= self.x_max
            assert y >= 0
            assert y <= self.y_max
        except AssertionError, e:
            print x, y
            raise e
        return x, y

    def animate(self, trajectory):
        frames = [self.new_image(fill=0), self.new_image(fill=1), self.new_image(fill=0), self.new_image(fill=1)]
        for i, (x, y) in enumerate(trajectory):
            frame = self.new_image(fill=1)
            if self.n_trace > 0:
                for n in reversed(range(self.n_trace)):
                    #                 try:
                    #                     x_, y_ = transform(*trajectory[i-n])
                    #                     frame[x_-delta_x:x_+delta_x, y_-delta_y:y_+delta_y] = 0.3 + n * .7 / n_trace
                    if i - n - 1 < 0:
                        continue
                    x_from, y_from = self.transform(*trajectory[i - n - 1])
                    x_to, y_to = self.transform(*trajectory[i - n])

                    xdiff = abs(x_to - x_from)
                    if xdiff == 0:
                        xdiff = .01
                    delta = (y_to - y_from) / xdiff
                    if x_from > x_to:
                        x_from, y_from, x_to, y_to = x_to, y_to, x_from, y_from
                    for n_line, x_ in enumerate(range(self.round(x_from), self.round(x_to)+1)):
                        p_x = x_
                        p_y = min(self.y_max - 1, self.round(y_from + delta * n_line))
                        frame[p_x-self.trace_width:p_x+self.trace_width,
                              p_y-self.trace_width:p_y+self.trace_width] = (
                        n * 1. / self.n_trace)
                        #                     np.fill_diagonal(frame[x_:x__, y_:y__], 0.3 + n * .7 / n_trace)
                        #                 except IndexError:
                        #                     pass
            x, y = self.transform(x, y)
            frame[x - self.hand_width:x + self.hand_width, y - self.hand_height:y + self.hand_height] = 0

            frames.append(frame)
        return frames

    def produce_selected_gifs(self):
        for item in self.lstSignals.selectedItems():
            self.render_single_gif(item.speaker, item.phase, item.meaning)

    def render_single_gif(self, participant=None, phase=None, image=None, traj=None, temp_file=None):
        if not traj:
            traj = self.data[participant][phase][image]
        if traj:
            data = map(lambda d: d.get_stabilized_position()[:2], traj)
            animation = self.animate(data)
            # print participant, phase, meaning
            if temp_file is None:
                meaning = image.split("(")[-1].split(".")[0]
                fname = os.path.join(self.txtOutputPath.text(), "%s_phase%s_%s.gif") % (participant, phase, meaning)
                print "Outputting %s..." % fname
                im.mimsave(fname, animation,
                       fps=self.fps, subrectangles=True)
            else:
                im.mimsave(temp_file, animation,
                           fps=self.fps, subrectangles=True)

    def log_to_gif(self, logfile, fps=10, n_trace=5):
        print "Reading %s..." % logfile
        r = fromFile(logfile)
        print "Animating trajectories..."
        self.n_trace = n_trace
        self.produce_selected_gifs(r, fps=fps)

if __name__ == '__main__':
    # app = QtGui.QApplication(sys.argv)
    from twisted.internet import reactor
    # console = embedded.EmbeddedIPython(app)#, main.window)
    # print "Embedded."
    import sys

    main = None
    # if len(sys.argv) > 1 and sys.argv[1] == "constantrate":
    #     main = ConstantRateRecorder()
    # else:
    main = PlotterWindow()
    main.show()
    # console.show()
    reactor.runReturn()
    sys.exit(app.exec_())
