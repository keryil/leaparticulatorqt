# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/res/learningWindow.ui'
#
# Created by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class Ui_learningWindow(object):
    def setupUi(self, learningWindow):
        learningWindow.setObjectName(_fromUtf8("learningWindow"))
        learningWindow.resize(832, 630)
        self.centralwidget = QtGui.QWidget(learningWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(800, 0))
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 40, 791, 221))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_5.addWidget(self.groupBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.btnPlay = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnPlay.sizePolicy().hasHeightForWidth())
        self.btnPlay.setSizePolicy(sizePolicy)
        self.btnPlay.setMinimumSize(QtCore.QSize(300, 30))
        self.btnPlay.setObjectName(_fromUtf8("btnPlay"))
        self.horizontalLayout_2.addWidget(self.btnPlay)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.btnRecord = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnRecord.sizePolicy().hasHeightForWidth())
        self.btnRecord.setSizePolicy(sizePolicy)
        self.btnRecord.setMinimumSize(QtCore.QSize(300, 30))
        self.btnRecord.setObjectName(_fromUtf8("btnRecord"))
        self.horizontalLayout_3.addWidget(self.btnRecord)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.btnSubmit = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSubmit.sizePolicy().hasHeightForWidth())
        self.btnSubmit.setSizePolicy(sizePolicy)
        self.btnSubmit.setMinimumSize(QtCore.QSize(300, 30))
        self.btnSubmit.setObjectName(_fromUtf8("btnSubmit"))
        self.horizontalLayout_4.addWidget(self.btnSubmit)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.volumeDial = QtGui.QDial(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.volumeDial.sizePolicy().hasHeightForWidth())
        self.volumeDial.setSizePolicy(sizePolicy)
        self.volumeDial.setMinimumSize(QtCore.QSize(100, 100))
        self.volumeDial.setMaximumSize(QtCore.QSize(100, 100))
        self.volumeDial.setMinimum(1)
        self.volumeDial.setMaximum(100)
        self.volumeDial.setInvertedAppearance(False)
        self.volumeDial.setInvertedControls(False)
        self.volumeDial.setNotchesVisible(True)
        self.volumeDial.setObjectName(_fromUtf8("volumeDial"))
        self.horizontalLayout.addWidget(self.volumeDial)
        self.verticalLayout.addLayout(self.horizontalLayout)
        learningWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(learningWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 832, 16))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        learningWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(learningWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        learningWindow.setStatusBar(self.statusbar)

        self.retranslateUi(learningWindow)
        QtCore.QMetaObject.connectSlotsByName(learningWindow)

    def retranslateUi(self, learningWindow):
        learningWindow.setWindowTitle(_translate("learningWindow", "Leap Theremin Experiment", None))
        self.groupBox.setTitle(_translate("learningWindow", "Create a signal to label this object", None))
        self.label.setText(_translate("learningWindow", "IMAGE", None))
        self.btnPlay.setText(_translate("learningWindow", "Play", None))
        self.btnRecord.setText(_translate("learningWindow", "Record", None))
        self.btnSubmit.setText(_translate("learningWindow", "Submit", None))
        self.volumeDial.setToolTip(_translate("learningWindow", "<html><head/><body><p>Volume</p></body></html>", None))

