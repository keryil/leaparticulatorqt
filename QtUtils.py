from PyQt4 import QtCore, QtGui, uic
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


class AbstractMeaning(object):
    # separates the feature values in the filename
    feature_sep = "-"
    def __init__(self, feature_dict,  feature_order):
        """
        feature_dict holds the features a specific implementation 
        may net such as size, color etc., and feature_order is a
        list representing the order of feature values while determining
        the filename.
        """
        assert set(feature_order) == set(feature_dict)
        self.feature_order = feature_order
        self.feature_dict = feature_dict

    def filename(self):
        """
        Generates the filename of the image dynamically using the 
        Constants module and OS-specific path settings of os.path
        module.
        """
        tuples = [str(self.feature_dict[f]) for f in self.feature_order]
        format_txt = self.feature_sep.join(["%s"] * len(tuples))
        format_txt += ".%s"
        # print format_txt
        # print tuples
        import sys;sys.stdout.flush()
        tuples.append(IMG_EXTENSION)
        fn = join(MEANING_DIR, format_txt %
                             tuple(tuples)) 
        assert path.isfile(join(getcwd(), fn))
        return fn

    def FromFile(filename):
        name = filename.split(sep)[-1].split('.')[0]
        args = map(int, feature_sep.splitlist(name))
        return Meaning(*args)

    def pixmap(self, tint=None):
        """
        Tint is a tuple (r,g,b,a).
        """
        base = QtGui.QPixmap(self.filename())
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

        return px

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features = [str(self.feature_dict[f]) for f in self.feature_order]
        s = self.feature_sep.join(features)
        return "%s(%s.%s)" % (self.__class__.__name__, s, IMG_EXTENSION)


class FeaturelessMeaning(AbstractMeaning):
    def __init__(self, id_no):
        super(FeaturelessMeaning, self).__init__(feature_dict={"id_no":id_no}, 
                                                 feature_order=['id_no'])
        from os.path import join
        assert 0 < id_no < 16
        self.id_no = id_no

class Meaning(object):

    def __init__(self, size, color, shade):
        from os.path import join
        for dim in (size, color, shade):
            assert 1 <= dim <= 6
        self.size = size
        self.color = color
        self.shade = shade
        
    def filename(self):
        """
        Generates the filename of the image dynamically using the 
        Constants module and OS-specific path settings of os.path
        module.
        """
        fn = join(MEANING_DIR, "%s%s%s.%s" %
                             (self.size, self.color, self.shade, IMG_EXTENSION)) 
        assert path.isfile(join(getcwd(), fn))
        return fn


    def FromFile(filename):
        name = filename.split(sep)[-1].split('.')[0]
        args = map(int, list(name))
        return Meaning(*args)

    def pixmap(self, tint=None):
        """
        Tint is a tuple (r,g,b,a).
        """
        base = QtGui.QPixmap(self.filename())
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

def loadUiWidget(uifilename,
               parent=None,
               root=getcwd()):
    """
    Convenience method to load and setup
    and widget, a QMainWindow by default.
    """
    ui_file = join(root, QT_DIR, uifilename)
    print "Loading ui file: %s" %ui_file
    w = uic.loadUi(ui_file)
    w.setParent(parent)
    return w

# def loadWidget(uifilename,
#                base_type=QtGui.QMainWindow,
#                parent=None):
#     """
#     Convenience method to load and setup
#     and widget, a QMainWindow by default.
#     """
#     widget = base_type(parent=parent)
#     ui = loadUiWidget(uifilename=uifilename)
#     print "Loaded widget..."
#     ui.setupUi(widget)
#     print "Setup widget."
#     return widget


# def loadUiWidget(uifilename, parent=None, root=getcwd()):
#     """
#     This function loads a QT widget from a .ui file.
#     """
#     ui_class = loadUiWidgetClass(uifilename, parent, root)
#     ui = ui_class()
#     return ui


# def loadUiWidgetClass(uifilename, parent=None, root=getcwd()):
#     """
#     This function loads a QT widget from a .ui file.
#     """
#     ui_file = join(root, QT_DIR, uifilename)
#     bare_name = basename(uifilename).split('.')[0]
#     py_file = open(join(root, QT_DIR, bare_name + '.py'), 'w')

#     print "Writing python module %s for file %s" % (py_file, ui_file)
#     compileUi(ui_file, py_file)
#     py_file.close()
#     module_name = bare_name
#     class_name = "Ui_%s" % module_name

#     ui_class = getattr(import_module("%s.%s" % (QT_DIR,
#                                                 module_name)),
#                        class_name)
#     print "Loaded the class %s" % class_name
#     return ui_class


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
