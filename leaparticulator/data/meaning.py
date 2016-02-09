import os
from os.path import join

from PyQt4 import QtGui, QtCore

from leaparticulator.constants import (IMG_EXTENSION, MEANING_DIR, TRUE_OVERLAY,
                                       FALSE_OVERLAY, MEANING_DIR_P2P, ROOT_DIR)

loadFromRes = lambda path: open(join(ROOT_DIR, "res", path + ".txt")).read()

class AbstractMeaning(object):
    # separates the feature values in the filename
    feature_sep = "-"

    def __init__(self, feature_dict, feature_order):
        """
        feature_dict holds the features a specific implementation 
        may net such as size, color etc., and feature_order is a
        list representing the order of feature values while determining
        the filename.
        """
        assert set(feature_order) == set(feature_dict)
        self.feature_order = feature_order
        self.feature_dict = feature_dict
        self.MEANING_DIR = MEANING_DIR

        # _ready holds the untinted QPixmap once it is generated
        self._ready = None

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
        import sys

        sys.stdout.flush()
        tuples.append(IMG_EXTENSION)
        fn = join(self.MEANING_DIR, format_txt %
                  tuple(tuples))
        # print join(getcwd(), fn)
        # print "Image path: ", join(ROOT_DIR, fn)
        assert os.path.isfile(join(ROOT_DIR, fn))
        return fn

    @classmethod
    def FromFile(cls, filename):
        assert os.path.isfile(filename)
        print "New meaning from: %s" % filename
        # print "Feature separator is: %s" % cls.feature_sep
        name = filename.split(os.path.sep)[-1].split('.')[0]
        args = name.split(cls.feature_sep)
        print args
        args = map(int, args)
        return cls(*args)

    def pixmap(self, tint=None, overlayed_text=None,
               font=QtGui.QFont("Arial", 27)):
        """
        Tint is a tuple (r,g,b,a).
        """
        # if tint is None and self._ready:
        #     return self._ready
        base = QtGui.QPixmap(self.filename())
        px = base
        # self._ready = px

        if tint is not None:
            px = QtGui.QPixmap(250, 250)
            px.fill(QtCore.Qt.transparent)
            r, g, b, a = tint
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
            painter.drawPixmap(0, 0, base)
            painter.drawPixmap(0, 0, overlay)
            painter.end()

        if overlayed_text:
            painter = QtGui.QPainter(px)
            painter.drawPixmap(0, 0, base)
            painter.setPen(QtCore.Qt.red)
            painter.setFont(font)
            painter.drawText(base.rect(),
                             QtCore.Qt.AlignCenter,
                             overlayed_text)
            painter.end()

        return px

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features = [str(self.feature_dict[f]) for f in self.feature_order]
        s = self.feature_sep.join(features)
        return "%s(%s.%s)" % (self.__class__.__name__, s, IMG_EXTENSION)

    def __eq__(self, other):
        return str(self) == str(other)


class P2PMeaning(AbstractMeaning):
    feature_sep = "_"

    def __init__(self, feature_dict, feature_order):
        super(P2PMeaning, self).__init__(feature_dict, feature_order)
        self.MEANING_DIR = MEANING_DIR_P2P

    @classmethod
    def FromFile(cls, filename):
        # print "Feature separator is: %s" % cls.feature_sep
        name = filename.split(os.path.sep)[-1].split('.')[0]
        args = name.split(cls.feature_sep)
        args = map(int, args)
        feature_dict = {"param%s" % i: arg for i, arg in enumerate(args)}
        feature_order = ["param%s" % i for i in range(len(args))]
        return cls(feature_dict, feature_order)


class FeaturelessMeaning(AbstractMeaning):
    def __init__(self, id_no):
        super(FeaturelessMeaning, self).__init__(feature_dict={"id_no": id_no},
                                                 feature_order=['id_no'])
        assert 0 < id_no < 16
        self.id_no = id_no


class Meaning(object):
    def __init__(self, size, color, shade):
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
        assert os.path.isfile(join(ROOT_DIR, fn))
        return fn

    def FromFile(filename):
        name = filename.split(os.path.sep)[-1].split('.')[0]
        args = map(int, list(name))
        return Meaning(*args)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Meaning(%s%s%s.%s)" % (self.size, self.color, self.shade, IMG_EXTENSION)
