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
from PyQt4.QtCore import Qt

import matplotlib

matplotlib.use("Qt4Agg")
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from QtUtils import connect, disconnect, loadUiWidget
from leaparticulator import constants
from leaparticulator.data.functions import fromFile
import matplotlib.pyplot as plt
import os
from itertools import product


class BrowserWindow(object):
    def __init__(self, parent=None):
        self.window = loadUiWidget(constants.BROWSER_UI)
        self.statusbar = self.get_child(QtGui.QStatusBar, 'statusbar')
        self.layout = self.get_child(QtGui.QVBoxLayout, 'verticalLayout')
        self.splitter = QtGui.QSplitter(Qt.Vertical)
        # self.splitter.setSizePolicy(QtGui.QSizePolicy.ExpandFlag)
        self.file_tree = QtGui.QTreeView()
        self.log_tree = QtGui.QTreeView()

        self.setup_matplotlib()
        self.setup_splitter()

        self.dir = os.path.join(constants.ROOT_DIR, "logs")
        self.dir_model = QtGui.QFileSystemModel()
        self.log_model = QtGui.QStandardItemModel()
        self.setup_file_model()

        # this is a dict of pandas dataframes, indexed by the log file
        self.data = {}

    def setup_matplotlib(self):
        # a figure instance to plot on
        self.figure = plt.figure()
        # return

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self.window)

        # Just some button connected to `plot` method
        self.chkMultivariate = QtGui.QCheckBox("Multivariate?")
        self.chkReversed = QtGui.QCheckBox("Reversed?")

        self.btnPlotTrajectory = QtGui.QPushButton('Plot trajectory')
        connect(self.btnPlotTrajectory, "clicked()", self.plot_trajectory)

        self.btnPlotHmmAndTrajectory = QtGui.QPushButton('Plot trajectory with HMM')
        connect(self.btnPlotHmmAndTrajectory, "clicked()", self.plot_hmm_and_trajectory)

        self.btnPlotHmm = QtGui.QPushButton('Plot HMM')
        connect(self.btnPlotHmm, "clicked()", self.plot_hmm)

    def setup_splitter(self):
        self.layout.addWidget(self.splitter)
        widgets = [self.file_tree, self.log_tree, self.toolbar, self.canvas, self.chkMultivariate,
                   self.chkReversed, self.btnPlotTrajectory, self.btnPlotHmmAndTrajectory, self.btnPlotHmm]
        [self.splitter.addWidget(w) for w in widgets]
        self.splitter.setStretchFactor(widgets.index(self.toolbar), 0)
        self.splitter.setStretchFactor(widgets.index(self.btnPlotHmm), 0)
        self.splitter.setStretchFactor(widgets.index(self.btnPlotTrajectory), 0)
        self.splitter.setStretchFactor(widgets.index(self.btnPlotHmmAndTrajectory), 0)

        # hard-wire no expansion
        widgets = [self.toolbar, self.btnPlotHmmAndTrajectory, self.btnPlotHmm, self.btnPlotTrajectory,
                   self.chkMultivariate, self.chkReversed]
        for w in widgets:
            w.setMinimumHeight(30)
            w.setMaximumHeight(30)

    def setup_file_model(self):
        print "Root folder is %s" % self.dir
        self.dir_model.setRootPath(self.dir)
        self.dir_model.setNameFilters(["*.exp.log"])

        self.file_tree.setModel(self.dir_model)
        self.file_tree.setRootIndex(self.dir_model.index(self.dir))

        self.file_tree_selection = self.file_tree.selectionModel()
        connect(self.file_tree_selection, "selectionChanged (const QItemSelection&,const QItemSelection&)",
                self.on_select_file)

        self.file_tree.setSortingEnabled(True)

    def update_status(self, string):
        self.statusbar.showMessage(str(string))
        from PyQt4.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        app.processEvents()


    def on_select_file(self, new, old):
        selected = new.indexes()[0]
        fname = self.dir_model.fileInfo(selected).absoluteFilePath()
        if fname.endswith("exp.log") and (fname not in self.data):
            self.update_status("Reading in log file %s" % fname)
            self.data[fname] = fromFile(fname)
            self.display_file_data(self.data[fname])
        else:
            return
        self.current_fname = fname
        self.update_status(self.data[fname])
        self.statusbar.showMessage("Done")

    def display_file_data(self, data):
        responses, test_results, responses_practice, test_results_practice, images = data
        responses = responses['127.0.0.1']
        responses_practice = responses_practice['127.0.0.1']
        phases = map(str, range(3))
        meanings = []
        for m in map(lambda x: responses[x].keys(), phases):
            meanings.extend(m)

        meanings = list(set(meanings))
        # meaning_items = []
        # for m in meanings:
        #     item = QtGui.QStandardItem()
        #     item.setText(str(m))
        #     meaning_items.append(item)
        # print meaning_items
        self.log_tree.setSortingEnabled(True)
        self.log_model = QtGui.QStandardItemModel()

        self.log_tree.setModel(self.log_model)
        root = self.log_model.invisibleRootItem()
        self.log_tree.setRootIndex(root.index())
        self.log_selection = self.log_tree.selectionModel()

        self.log_model.setColumnCount(4)
        self.log_model.setHorizontalHeaderLabels(['Phase', 'Meaning', 'Practice', 'Duration'])

        for rr in (responses, responses_practice):
            for meaning, phase in product(meanings, phases):
                if meaning not in rr[phase]:
                    continue
                vals = [str(phase),
                        str(meaning),
                        str(rr == responses_practice),
                        str(len(rr[phase][meaning]))]

                row = [QtGui.QStandardItem(),
                       QtGui.QStandardItem(),
                       QtGui.QStandardItem(),
                       QtGui.QStandardItem()]

                for v, r in zip(vals, row):
                    r.setData(rr[phase][meaning])
                    r.setText(v)

                # row[-1].setData(rr[phase][meaning])

                self.log_model.appendRow(row)

    def get_child(self, qtype, qname):
        obj = self.window.findChildren(qtype, qname)
        if len(obj):
            return obj[0]
        else:
            return None

    def plot_quiver2d(self, data, axis, alpha=.75, C=[], path=None, *args, **kwargs):
        X, Y = zip(*data[:-1])
        u = [x1 - x0 for x0, x1 in zip(X[:-1], X[1:])]
        v = [y1 - y0 for y0, y1 in zip(Y[:-1], Y[1:])]
        if not C:
            color_delta = 1. / (len(X) - 1)
            C = [(color_delta * i, color_delta * i, color_delta * i) for i in range(len(X) - 1)]
        X = X[:-1]
        Y = Y[:-1]
        print map(len, [X, Y, u, v])
        patches = axis.quiver(X, Y, u, v, C, scale_units='xy', angles='xy', scale=1, width=0.005, alpha=alpha, **kwargs)
        return patches

    def plot_trajectory(self):
        ''' plot some random stuff '''
        reversed = self.chkReversed.isChecked()
        multivariate = self.chkMultivariate.isChecked()

        indexes = self.log_selection.selectedRows()
        if not indexes:
            return
        index = indexes[0]
        item = self.log_model.itemFromIndex(index)
        print "Item:", item
        data = None

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        if multivariate:
            data = [f.get_stabilized_position()[:2] for f in item.data()]
            if reversed:
                data = [(d1, d0) for d0, d1 in data]
            # data0, data1 = zip(*data)
            # ax.plot(data0, data1, '->')
            self.plot_quiver2d(data, ax)
            if reversed:
                ax.set_xlabel("Y-coordinate")
                ax.set_xlabel("X-coordinate")
            else:
                ax.set_xlabel("X-coordinate")
                ax.set_xlabel("Y-coordinate")
        else:
            if not reversed:
                data = [f.get_stabilized_position()[0] for f in item.data()]
            else:
                data = [f.get_stabilized_position()[1] for f in item.data()]
            self.plot_quiver2d([d for d in enumerate(data)], ax)
            # ax.plot(data, '->')
            ax.set_xlabel("Time")
            if reversed:
                ax.set_ylabel("X-coordinate")
            else:
                ax.set_ylabel("Y-coordinate")

        print "Data:", data

        # refresh canvas
        self.canvas.draw()

    def plot_hmm_and_trajectory(self):
        indexes = self.log_selection.selectedRows()
        print map(lambda x: x.text(), map(self.log_model.itemFromIndex, indexes))

    def plot_hmm(self):
        from leaparticulator.notebooks.StreamlinedDataAnalysisGhmm import plot_hmm, unpickle_results
        indexes = self.log_selection.selectedRows()
        phase = self.log_model.itemFromIndex(indexes[0]).text()

        # which phase are we interested in?

        # first, ask the user which HMM they want to plot
        import fnmatch
        fname_to_logid = lambda x: os.path.split(x)[-1].split(".exp.log")[0]
        print self.current_fname, fname_to_logid(self.current_fname)
        matches = []
        base_id = fname_to_logid(self.current_fname)
        query = '%s*phase%d*.hmms' % (base_id, int(phase))
        print "Searching for filename format: %s" % query
        self.update_status("Searching %s recursively for HMM files associated with this trajectory..." % self.dir)
        for root, dirnames, filenames in os.walk(self.dir, followlinks=True):
            for filename in fnmatch.filter(filenames, query):
                matches.append(os.path.join(root, filename))
        hmm_file = self.pick_hmm_file(matches)

        # unpickle the HMMs
        self.update_status("Unpickling the HMMs...")
        results = unpickle_results(hmm_file)
        hmm = results.hmms[0]
        for hmm_ in results.hmms:
            if hmm.bic > hmm.bic:
                hmm = hmm_

        # now for the actual plotting
        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        plot_hmm(hmm.means, hmm.transmat, hmm.variances, hmm.initProb, axes=ax)

    def pick_hmm_file(self, filenames):
        dialog = loadUiWidget("HmmFilePickerDialog.ui")
        lst = dialog.findChildren(QtGui.QListWidget, "listWidget")[0]
        lst.addItems(filenames)
        dialog.exec_()
        return lst.currentItem().text()

    def show(self):
        self.window.show()

# class PickHMMDialog(QtGui.QDialog):
#     def __init__(self, filelist, parent=None):
#         super(PickHMMDialog, self).__init__(parent)
#         self.list = QtGui.QListWidget(self)
#         self.list.addItems(filelist)
#         self.buttons = QtGui.QDialogButtonBox(
#             QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
#             Qt.Horizontal, self)
#         self.buttons.accepted.connect(self.accept)
#         self.buttons.rejected.connect(self.reject)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # console = embedded.EmbeddedIPython(app)#, main.window)
    # print "Embedded."
    main = BrowserWindow()
    main.show()
    # console.show()
    sys.exit(app.exec_())
