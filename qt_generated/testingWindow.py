# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/qt_generated/testingWindow.ui'
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

class Ui_testingWindow(object):
    def setupUi(self, testingWindow):
        testingWindow.setObjectName(_fromUtf8("testingWindow"))
        testingWindow.resize(1069, 712)
        self.centralwidget = QtGui.QWidget(testingWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setStyleSheet(_fromUtf8("background-color: white"))
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.pushButton_3 = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setMinimumSize(QtCore.QSize(300, 300))
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setStyleSheet(_fromUtf8("background-color: white; color: white"))
        self.pushButton_3.setIconSize(QtCore.QSize(100, 100))
        self.pushButton_3.setCheckable(True)
        self.pushButton_3.setAutoExclusive(True)
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.gridLayout.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.pushButton_1 = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pushButton_1.sizePolicy().hasHeightForWidth())
        self.pushButton_1.setSizePolicy(sizePolicy)
        self.pushButton_1.setMinimumSize(QtCore.QSize(300, 300))
        self.pushButton_1.setAutoFillBackground(False)
        self.pushButton_1.setStyleSheet(_fromUtf8("background-color: white; color: white"))
        self.pushButton_1.setIconSize(QtCore.QSize(100, 100))
        self.pushButton_1.setCheckable(True)
        self.pushButton_1.setAutoExclusive(True)
        self.pushButton_1.setFlat(True)
        self.pushButton_1.setObjectName(_fromUtf8("pushButton_1"))
        self.gridLayout.addWidget(self.pushButton_1, 0, 0, 1, 1)
        self.pushButton_4 = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy)
        self.pushButton_4.setMinimumSize(QtCore.QSize(300, 300))
        self.pushButton_4.setAutoFillBackground(False)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: white; color: white"))
        self.pushButton_4.setIconSize(QtCore.QSize(100, 100))
        self.pushButton_4.setCheckable(True)
        self.pushButton_4.setAutoExclusive(True)
        self.pushButton_4.setFlat(True)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.gridLayout.addWidget(self.pushButton_4, 1, 1, 1, 1)
        self.pushButton_2 = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setMinimumSize(QtCore.QSize(300, 300))
        self.pushButton_2.setAutoFillBackground(False)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: white; color: white"))
        self.pushButton_2.setIconSize(QtCore.QSize(100, 100))
        self.pushButton_2.setCheckable(True)
        self.pushButton_2.setAutoExclusive(True)
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.horizontalLayout_3.addWidget(self.groupBox)
        self.frame = QtGui.QFrame(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(370, 0))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayoutWidget = QtGui.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 0, 271, 471))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setContentsMargins(-1, 100, -1, -1)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.btnPlay = QtGui.QPushButton(self.verticalLayoutWidget)
        self.btnPlay.setObjectName(_fromUtf8("btnPlay"))
        self.verticalLayout.addWidget(self.btnPlay)
        self.btnSubmit = QtGui.QPushButton(self.verticalLayoutWidget)
        self.btnSubmit.setObjectName(_fromUtf8("btnSubmit"))
        self.verticalLayout.addWidget(self.btnSubmit)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.volumeDial = QtGui.QDial(self.verticalLayoutWidget)
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
        self.horizontalLayout_4.addWidget(self.volumeDial)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3.addWidget(self.frame)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        testingWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(testingWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        testingWindow.setStatusBar(self.statusbar)

        self.retranslateUi(testingWindow)
        QtCore.QMetaObject.connectSlotsByName(testingWindow)

    def retranslateUi(self, testingWindow):
        testingWindow.setWindowTitle(_translate("testingWindow", "Leap Theremin Experiment", None))
        self.groupBox.setTitle(_translate("testingWindow", "Choose the object you think the signal represents.", None))
        self.pushButton_3.setText(_translate("testingWindow", "PushButton", None))
        self.pushButton_3.setShortcut(_translate("testingWindow", "Ctrl+S", None))
        self.pushButton_1.setText(_translate("testingWindow", "PushButton", None))
        self.pushButton_1.setShortcut(_translate("testingWindow", "Ctrl+S", None))
        self.pushButton_4.setText(_translate("testingWindow", "PushButton", None))
        self.pushButton_4.setShortcut(_translate("testingWindow", "Ctrl+S", None))
        self.pushButton_2.setText(_translate("testingWindow", "PushButton", None))
        self.pushButton_2.setShortcut(_translate("testingWindow", "Ctrl+S", None))
        self.btnPlay.setText(_translate("testingWindow", "Play", None))
        self.btnSubmit.setText(_translate("testingWindow", "Submit", None))
        self.volumeDial.setToolTip(_translate("testingWindow", "<html><head/><body><p>Volume</p></body></html>", None))

