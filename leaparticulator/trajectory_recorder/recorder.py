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

        self.setLogFile(self.filename)
        self.score = self.data['1']['./img/meanings/1_1.png']
        import os
        self.txtOutputPath.setText(os.path.abspath(os.path.expanduser("~" + os.sep + "Desktop")))

        self.actionPlay.triggered.connect(self.play)
        self.actionRecord.triggered.connect(self.record)
        self.actionChange_Path.triggered.connect(self.setOutputPath)
        self.score = None

        print "Done!"

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
        thereminplayback.start(item.signal,
                               filename=os.path.join(self.txtOutputPath.text(), "phase%s-%s.wav" % (item.phase,
                                                                                                    os.path.split(
                                                                                                        item.meaning)[
                                                                                                        -1].split(".")[
                                                                                                        0])),
                               jsonencoded=False,
                               callback=lambda: reactor.iterate(1000) and callback())

    def record(self):
        try:
            self.count += 1
            print "Outputting item %d (p%s/%s)..." % (self.count, self.items[self.count].phase,
                                                      self.items[self.count].meaning)
        except:
            print "Outputting %d items!" % len(self.items)
            self.count = 0
        try:
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
