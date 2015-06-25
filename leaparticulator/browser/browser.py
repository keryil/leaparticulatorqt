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
        self.mdi = self.get_child(QtGui.QMdiArea, 'mdiArea')
        self.statusbar = self.get_child(QtGui.QStatusBar, 'statusbar')

        self.setup_file_dock()

        # connect new plot action
        action = self.get_child(QtGui.QAction, 'actionNew_plot')
        print "Action:", action
        connect(action, action.triggered, self.new_figure)
        action.trigger()
        action.trigger()
        print self.figures()
        # self.setup_splitter()
        
        self.dir = os.path.join(constants.ROOT_DIR, "logs")
        self.dir_model = QtGui.QFileSystemModel()
        self.log_model = QtGui.QStandardItemModel()
        self.setup_file_model()
        
        # this is a dict of pandas dataframes, indexed by the log file
        self.data = {}

        self.mdi.tileSubWindows()

    def setup_file_dock(self):
        # this will hold the tree views
        tree_splitter = QtGui.QSplitter(Qt.Vertical)
        self.file_tree = QtGui.QTreeView()
        self.log_tree = QtGui.QTreeView()
        tree_splitter.addWidget(self.file_tree)
        tree_splitter.addWidget(self.log_tree)

        self.file_dock = QtGui.QDockWidget("Files/Trajectories")
        self.file_dock.setWidget(tree_splitter)
        self.window.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)

    def figures(self):
        return [w.windowTitle() for w in self.mdi.subWindowList()]
    
    def new_figure(self):
        # a frame to hold everything
        container = QtGui.QFrame()
        canvas_win = self.mdi.addSubWindow(container)

        plot_id = "Plot #%d" % len(self.mdi.subWindowList())
        print plot_id

        canvas_win.setWindowTitle(plot_id)

        # a figure instance to plot on
        container.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        container.canvas = FigureCanvas(container.figure)
        
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        toolbar = NavigationToolbar(container.canvas, container)
        
        # Just some button connected to `plot` method
        container.chkMultivariate = QtGui.QCheckBox("Multivariate?")
        container.chkReversed = QtGui.QCheckBox("Reversed?")
        
        container.btnPlotTrajectory = QtGui.QPushButton('Plot trajectory')
        connect(container.btnPlotTrajectory, "clicked()", lambda: self.plot_trajectory(container))
        
        container.btnPlotHmmAndTrajectory = QtGui.QPushButton('Plot trajectory with HMM')
        connect(container.btnPlotHmmAndTrajectory, "clicked()", lambda: self.plot_hmm_and_trajectory(container))
        
        container.btnPlotHmm = QtGui.QPushButton('Plot HMM')
        connect(container.btnPlotHmm, "clicked()", lambda: self.plot_hmm(container))

        layout = QtGui.QVBoxLayout()
        container.setLayout(layout)
        [layout.addWidget(w) for w in [toolbar, container.canvas, container.chkMultivariate,
                   container.chkReversed, container.btnPlotTrajectory, container.btnPlotHmmAndTrajectory,
                                       container.btnPlotHmm]]
        print "Finished %s" % plot_id
        canvas_win.show()

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

                self.log_model.appendRow(row)

    def get_child(self, qtype, qname):
        """

        :QWidget : child
        """
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

    def plot_trajectory(self, container=None):
        ''' plot some random stuff '''
        if not container:
            container = self
        reversed = container.chkReversed.isChecked()
        multivariate = container.chkMultivariate.isChecked()

        indexes = self.log_selection.selectedRows()
        if not indexes:
            return
        index = indexes[0]
        item = self.log_model.itemFromIndex(index)
        print "Item:", item
        data = None

        # create an axis
        ax = container.figure.add_subplot(111)

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
                print "Frames: %s" % item.data()
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
        container.canvas.draw()

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

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # console = embedded.EmbeddedIPython(app)#, main.window)
    # print "Embedded."
    main = BrowserWindow()
    main.show()
    # console.show()
    sys.exit(app.exec_())
