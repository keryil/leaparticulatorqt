from os.path import join
from os import getcwd
from collections import defaultdict

from PyQt4 import QtCore, QtGui, uic

from leaparticulator.constants import QT_DIR, TESTING, ROOT_DIR


slots = defaultdict(list)
loadFromRes = lambda path: open(join(getcwd(), "res", path + ".txt")).read()


def setButtonIcon(button, pixmap):
    """
    Sets the icon of a QPushButton to the given pixmap, and 
    removes any text.
    """
    button.setIcon(QtGui.QIcon(pixmap))
    if pixmap is not None:
        button.setIconSize(pixmap.rect().size())
    button.setText("")


def loadUiWidget(uifilename,
                 parent=None,
                 root=ROOT_DIR,
                 widget=None):
    """
    Convenience method to load and setup
    and widget, a QMainWindow by default.
    """
    if TESTING:
        import os

        root = os.path.expanduser("~/Dropbox/ABACUS/Workspace/LeapArticulatorQt")
    ui_file = join(root, QT_DIR, uifilename)
    print "Loading ui file: %s" % ui_file
    w = uic.loadUi(ui_file, widget)
    w.setParent(parent)
    return w


loadWidget = loadUiWidget


def connect(widget, signal, slot, old_syntax=False, widget2=None):
    if isinstance(signal, str) or isinstance(slot, str):
        # print "Signal is string - using old style signals/slots"
        old_syntax = True
        if isinstance(slot, str):
            assert slot[-1] == ")"
        assert signal[-1] == ")"

    if not old_syntax:
        signal.connect(slot)
    else:
        if widget2 is None:
            QtCore.QObject.connect(widget, QtCore.SIGNAL(signal), slot)
        else:
            assert isinstance(slot, str)
            QtCore.QObject.connect(widget, QtCore.SIGNAL(signal), widget2, QtCore.SLOT(slot))

    if signal not in slots[widget]:
        slots[widget].append((signal, slot, old_syntax, widget2))


def disconnect(widget, slot=None):
    for signal, slot_, old_syntax, widget2 in slots[widget]:
        if slot is not None:
            if slot_ != slot:
                continue
        try:
            if old_syntax:
                if widget2 is None:
                    QtCore.QObject.disconnect(widget, QtCore.SIGNAL(signal), slot_)
                else:
                    QtCore.QObject.connect(widget, QtCore.SIGNAL(signal), widget2, QtCore.SLOT(slot_))
            else:
                signal.disconnect(slot)
        except RuntimeError, err:
            print "RuntimeError while disconnecting signal: %s" % err
