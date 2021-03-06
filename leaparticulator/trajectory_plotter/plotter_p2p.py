__author__ = 'kerem'
import os
import sip
import sys

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

from leaparticulator.oldstuff.QtUtils import loadUiWidget

from leaparticulator.data.functions import fromFile
import numpy as np
import imageio as im


class PlotterWindow(QtGui.QMainWindow):
    def __init__(self):
        super(PlotterWindow, self).__init__()
        loadUiWidget('Traj2GIF.ui', widget=self)
        self.setWindowTitle("Plotter")
        self.extension = None
        self.previewImage = None
        self.filename = None
        self.trajectory = None

        self.set_extension()
        self.txtOutputPath.setText(os.path.abspath(os.path.expanduser("~" + os.sep + "Desktop")))

        self.ignoreX = False
        self.ignoreY = False

        self.x_offset = 210
        self.y_max = 608
        self.x_max = 608
        self.fps = 150
        self.n_trace = 15
        self.trace_width = 2
        self.hand_width = 10
        self.hand_height = 10
        # number of black frames at the beginning of each video
        self.anchorCount = 4
        self.data, self.meanings = None, None

        self.setupSignals()

        print "Done!"

    def setupSignals(self):
        self.actionChangeExtension.triggered.connect(self.set_extension)
        self.actionPlay.triggered.connect(self.preview)
        self.actionRecord.triggered.connect(self.produce_selected_gifs)
        self.actionChange_Path.triggered.connect(self.setOutputPath)
        self.lstSignals.currentItemChanged.connect(self.selectTrajectory)

        self.spinPlayback.valueChanged.connect(self.setRate)
        self.spinAnchorFrames.valueChanged.connect(self.setAnchorCount)
        self.spinHandSize.valueChanged.connect(self.setHandSize)

        # fix the terribly slow open file dialog under
        # opensuse
        import platform
        options = QtGui.QFileDialog.Options(0)
        if platform.system() == "Linux":
            options = QtGui.QFileDialog.DontUseNativeDialog
        self.actionOpenLog.triggered.connect(
            lambda: self.setLogFile(str(QtGui.QFileDialog.getOpenFileName(options=options))))

        def toggle_x():
            self.ignoreX = not self.ignoreX
            print "Ignore x?", self.ignoreX

        def toggle_y():
            self.ignoreY = not self.ignoreY
            print "Ignore y?", self.ignoreY

        self.chkIgnoreX.stateChanged.connect(toggle_x)
        self.chkIgnoreY.stateChanged.connect(toggle_y)

        def set_ntrace(n):
            self.n_trace = n
            print "n_trace set to", n

        def set_trace_width(n):
            self.trace_width = n
            print "Trace width set to", n

        self.spinTrace.valueChanged.connect(set_ntrace)
        self.spinWidth.valueChanged.connect(set_trace_width)

    def updatePredictedLength(self):
        self.lblLength.setText("Predicted video duration is %.2f seconds for the last selected item." %
                               ((len(self.trajectory) + self.anchorCount) / float(self.fps)))

    def set_extension(self):
        self.extension = self.cmbExtension.currentText()
        self.btnRecord.setText("Render all selected items as %s" % (self.extension.upper()))
        print "Extension set to %s" % self.extension

    def setAnchorCount(self, count):
        self.anchorCount = count
        print "Number of anchor frames set to %s" % self.anchorCount
        self.updatePredictedLength()

    def setHandSize(self, size):
        self.hand_height = size
        self.hand_width = size
        print "Hand size set to %s" % size

    def setRate(self, rate):
        print "FPS set to", rate
        self.fps = rate
        self.updatePredictedLength()
        # self.previewImage.setSpeed(self.fps)

    def selectTrajectory(self, current, previous):
        if current:
            self.setTrajectory(current.signal)

    def setTrajectory(self, trajectory):
        self.trajectory = trajectory
        self.updatePredictedLength()

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
                            "Speaker %s, Phase %s, Meaning %s (%d Frames)" % (speaker.split("@")[0],
                                                                              phase,
                                                                              meaning,
                                                                              len(signal)))
                        item.signal = signal
                        item.meaning = meaning
                        item.phase = phase
                        item.speaker = speaker
                        self.items.append(item)
                        self.lstSignals.addItem(item)

    def preview(self):
        print "Preview!"
        f = "temp_file.%s" % self.extension
        self.render_single_gif(traj=self.trajectory, temp_file=f)
        self.previewImage = QtGui.QMovie(f)
        self.previewImage.setCacheMode(QtGui.QMovie.CacheAll)

        lblPreview = self.findChild(QtGui.QLabel, "lblPreview")
        lblPreview.setMovie(self.previewImage)
        self.previewImage.setSpeed(100)
        self.previewImage.start()

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
        if x == 0:
            x = self.x_max
        if y == 0:
            y = self.y_max

        data = np.empty((x, y))
        data.fill(fill)
        return data

    def transform(self, x, y, rounded=False):
        """
        Transforms given Leap coordinates to GIF coordinates.
        :param rounded:
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
        if rounded:
            x, y = round(x), round(y)
        if self.ignoreX:
            x = self.x_max / 2
        if self.ignoreY:
            y = self.y_max / 2
        return y, x

    def animate(self, trajectory):
        """
        Animates a trajectory and returns the frames. Note that inside the
        function all x's are actually y's and vice versa, since these are
        swapped at self.transform to work around the row first notation
        of numpy images I started out with.
        :param trajectory:
        :return:
        """
        frames = [self.new_image(fill=0)] * self.anchorCount
        # minimum_x = min(t[0] for t in trajectory)
        # if minimum_x < 0:
        #     self.x_offset = -minimum_x
        # else:
        #     self.x_offset = 0
        # print "X offset set to %s" % self.x_offset

        for i, (x, y) in enumerate(map(lambda x: self.transform(*x, rounded=True), trajectory)):
            frame = self.new_image(fill=1)
            if self.n_trace > 0:
                for n in reversed(range(1, self.n_trace + 1)):
                    if i - n - 1 < 0:
                        continue
                    x_from, y_from = self.transform(*trajectory[i - n - 1])
                    x_to, y_to = self.transform(*trajectory[i - n])

                    xdiff = abs(x_to - x_from)
                    if xdiff == 0:
                        if self.ignoreY:
                            down = True
                            if y_from > y_to:
                                y_from, y_to = y_to, y_from
                                down = False
                            for n_line, y_ in enumerate(range(round(y_from, down),
                                                              round(y_to, down) + 1)):
                                p_y = y_
                                p_x = x_to
                                frame[p_x - self.trace_width:p_x + self.trace_width,
                                p_y - self.trace_width:p_y + self.trace_width] = (
                                    n * 1. / self.n_trace)
                        continue
                    else:
                        delta = (y_to - y_from) / xdiff
                    down = True
                    if x_from > x_to:
                        x_from, x_to = x_to, x_from
                        down = False
                    for n_line, x_ in enumerate(range(round(x_from, down), round(x_to, down) + 1)):
                        p_x = x_
                        p_y = min(self.y_max - 1, round(y_from + delta * n_line))
                        frame[p_x - self.trace_width:p_x + self.trace_width,
                        p_y - self.trace_width:p_y + self.trace_width] = (n * 1. / self.n_trace)
            frame[x - self.hand_width:x + self.hand_width,
            y - self.hand_height:y + self.hand_height] = 0
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
            print "Rendering at %s FPS" % self.fps
            # print participant, phase, meaning
            if temp_file is None:
                meaning = image.split("(")[-1].split(".")[0]
                fname = os.path.join(self.txtOutputPath.text(), "%s_phase%s_%s.%s") % \
                        (participant, phase, meaning, self.extension)
                print "Outputting %s..." % fname
                im.mimsave(fname, animation,
                           fps=self.fps)
            else:
                im.mimsave(temp_file, animation,
                           fps=self.fps)

def round(flt, down=False):
    if down:
        return int(np.ceil(flt))
    return int(np.round(flt))

if __name__ == '__main__':
    import sys

    main = PlotterWindow()
    main.show()
    sys.exit(app.exec_())
