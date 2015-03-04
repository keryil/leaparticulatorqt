from PyQt4 import QtCore, QtGui
from PyQt4.uic import compileUi
from PyQt4.QtCore import QFile
from os.path import join, basename
from os import getcwd, sep, path
from importlib import import_module
from Constants import QT_DIR, MEANING_DIR, IMG_EXTENSION, TRUE_OVERLAY, FALSE_OVERLAY
from collections import defaultdict, namedtuple


slots = defaultdict(list)
# Meaning = namedtuple("Meaning", ["pixmap", "size", "color", "shade"])
loadFromRes = lambda path: open(join(getcwd(), "res", path + ".txt")).read()


class Meaning(object):

    def __init__(self, size, color, shade):
        from os.path import join
        for dim in (size, color, shade):
            assert 1 <= dim <= 6
        self.size = size
        self.color = color
        self.shade = shade
        self.filename = join(MEANING_DIR, "%s%s%s.%s" %
                             (size, color, shade, IMG_EXTENSION))
        # print join(getcwd(), self.filename)
        assert path.isfile(join(getcwd(), self.filename))

    def FromFile(filename):
        name = filename.split(sep)[-1].split('.')[0]
        args = map(int, list(name))
        return Meaning(*args)

    def pixmap(self, tint=None):
        """
        Tint is a tuple (r,g,b,a).
        """
        base = QtGui.QPixmap(self.filename)
        px = base
        if tint is not None:
            px = QtGui.QPixmap(250,250)
            px.fill(QtCore.Qt.transparent)
            r,g,b,a = tint
            correct = g > 1
            overlay = None
            if correct:
                overlay = QtGui.QPixmap(TRUE_OVERLAY)
                print "Overlay: %s" % TRUE_OVERLAY
            else:
                overlay = QtGui.QPixmap(FALSE_OVERLAY)
                print "Overlay: %s" % TRUE_OVERLAY
            # color = QtGui.QColor(r,g,b,a)
            painter = QtGui.QPainter(px)
            painter.drawPixmap(0,0,base)
            painter.drawPixmap(0,0,overlay)
            painter.end()
            # painter.setCompositionMode(painter.CompositionMode_Multiply)
            # painter.fillRect(px.rect(), color)

        return px

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Meaning(%s%s%s.%s)" % (self.size, self.color, self.shade, IMG_EXTENSION)

def setButtonIcon(button, pixmap):
    """
    Sets the icon of a QPushButton to the given pixmap, and 
    removes any text.
    """
    button.setIcon(QtGui.QIcon(pixmap))
    if pixmap is not None:
        button.setIconSize(pixmap.rect().size())
    button.setText("")

def loadWidget(uifilename,
               base_type=QtGui.QMainWindow,
               parent=None):
    """
    Convenience method to load and setup
    and widget, a QMainWindow by default.
    """
    widget = base_type(parent=parent)
    ui = loadUiWidget(uifilename=uifilename)
    ui.setupUi(widget)
    return widget


def loadUiWidget(uifilename, parent=None, root=getcwd()):
    """
    This function loads a QT widget from a .ui file.
    """
    ui_class = loadUiWidgetClass(uifilename, parent, root)
    ui = ui_class()
    return ui


def loadUiWidgetClass(uifilename, parent=None, root=getcwd()):
    """
    This function loads a QT widget from a .ui file.
    """
    ui_file = join(root, QT_DIR, uifilename)
    bare_name = basename(uifilename).split('.')[0]
    py_file = open(join(root, QT_DIR, bare_name + '.py'), 'w')

    print "Writing python module %s for file %s" % (py_file, ui_file)
    compileUi(ui_file, py_file)
    py_file.close()
    module_name = bare_name
    class_name = "Ui_%s" % module_name

    ui_class = getattr(import_module("%s.%s" % (QT_DIR,
                                                module_name)),
                       class_name)
    return ui_class


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
