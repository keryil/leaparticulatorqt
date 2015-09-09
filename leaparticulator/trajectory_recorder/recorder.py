__author__ = 'kerem'
import sys
import sip

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

from PyQt4 import QtGui, uic
from PyQt4.QtCore import Qt

from QtUtils import connect, disconnect, loadUiWidget

from leaparticulator.data.functions import fromFile
from leaparticulator.theremin.theremin import ThereminPlayback


class RecorderWindow(QtGui.QMainWindow):
    def __init__(self):
        super(RecorderWindow, self).__init__()
        # self.filename = "/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/leaparticulator/test/test_data/123R0126514.1r.exp.log"
        self.filename = None
        self.playback = ThereminPlayback(record=False)
        loadUiWidget('Traj2MP3.ui', widget=self)

        # self.setLogFile(self.filename)
        # self.score = self.data['1']['./img/meanings/1_1.png']
        import os
        self.txtOutputPath.setText(os.path.abspath(os.path.expanduser("~" + os.sep + "Desktop")))

        self.actionPlay.triggered.connect(self.play)
        self.actionRecord.triggered.connect(self.record)
        self.actionChange_Path.triggered.connect(self.setOutputPath)
        self.actionOpenLog.triggered.connect(lambda: self.setLogFile(str(QtGui.QFileDialog.getOpenFileName())))
        self.score = None

        self.lstSignals.currentItemChanged.connect(self.selectScore)

        print "Done!"

    def selectScore(self, current, previous):
        self.setScore(current.signal)

    def setScore(self, score):
        self.score = score

    def setOutputPath(self):
        print self.txtOutputPath.text()
        text = str(QtGui.QFileDialog.getExistingDirectory(None))
        self.txtOutputPath.setText(text)

    def setLogFile(self, fname):
        self.data = fromFile(fname)[0]['127.0.0.1']
        self.filename = fname
        self.populateWithSignals()
        self.btnPlay.setEnabled(True)
        self.btnRecord.setEnabled(True)

    def populateWithSignals(self):
        self.items = []
        self.lstSignals.clear()
        for phase in self.data:
            for meaning in self.data[phase]:
                item = QtGui.QListWidgetItem("Phase %s, Meaning %s" % (phase, meaning))
                item.signal = self.data[phase][meaning]
                item.meaning = meaning
                item.phase = phase
                self.items.append(item)
                self.lstSignals.addItem(item)

    def play(self):
        print "Play!"
        selected = self.lstSignals.selectedItems()
        if not selected:
            return
        selected = selected[0]
        self.playback.start(selected.signal, filename="/home/kerem/Desktop/ThereminSignal.wav", jsonencoded=False)

    def _record(self, thereminplayback, item, callback=None):
        import os
        from twisted.internet import reactor
        meaning = str(item.meaning)
        if os.sep in meaning:
            meaning = os.path.split(item.meaning)[-1].split(".")[0]
        thereminplayback.start(item.signal,
                               filename=os.path.join(self.txtOutputPath.text(), "phase%s-%s.wav" % (item.phase,
                                                                                                    meaning)),
                               jsonencoded=False,
                               callback=lambda: reactor.iterate(10) or callback())

    def record(self):
        try:
            self.count += 1
            print "Outputting item %d (p%s/%s)..." % (self.count, self.items[self.count].phase,
                                                      self.items[self.count].meaning)
        except:
            print "Outputting %d items!" % len(self.items)
            self.count = 0
        try:
            from twisted.internet import reactor
            reactor.iterate(100)
            item = self.items[self.count]
            playback = ThereminPlayback(record=True)
            self._record(playback, item, callback=self.record)
            # reactor.callLater(.5, self.record)
        except IndexError:
            print "Finished outputting all items!"


if __name__ == '__main__':
    # app = QtGui.QApplication(sys.argv)
    from twisted.internet import reactor
    # console = embedded.EmbeddedIPython(app)#, main.window)
    # print "Embedded."
    main = RecorderWindow()
    main.show()
    # console.show()
    reactor.runReturn()
    sys.exit(app.exec_())
