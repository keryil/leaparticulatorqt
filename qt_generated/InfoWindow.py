# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/qt_generated/InfoWindow.ui'
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

class Ui_InfoWindow(object):
    def setupUi(self, InfoWindow):
        InfoWindow.setObjectName(_fromUtf8("InfoWindow"))
        InfoWindow.resize(800, 729)
        self.centralwidget = QtGui.QWidget(InfoWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.lblInfo = QtGui.QLabel(self.centralwidget)
        self.lblInfo.setMaximumSize(QtCore.QSize(750, 16777215))
        self.lblInfo.setWordWrap(True)
        self.lblInfo.setObjectName(_fromUtf8("lblInfo"))
        self.horizontalLayout_2.addWidget(self.lblInfo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.lblMeaningSpace = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblMeaningSpace.sizePolicy().hasHeightForWidth())
        self.lblMeaningSpace.setSizePolicy(sizePolicy)
        self.lblMeaningSpace.setMinimumSize(QtCore.QSize(600, 500))
        self.lblMeaningSpace.setMaximumSize(QtCore.QSize(750, 16777215))
        self.lblMeaningSpace.setText(_fromUtf8(""))
        self.lblMeaningSpace.setScaledContents(False)
        self.lblMeaningSpace.setAlignment(QtCore.Qt.AlignCenter)
        self.lblMeaningSpace.setObjectName(_fromUtf8("lblMeaningSpace"))
        self.horizontalLayout_4.addWidget(self.lblMeaningSpace)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.btnOkay = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnOkay.sizePolicy().hasHeightForWidth())
        self.btnOkay.setSizePolicy(sizePolicy)
        self.btnOkay.setMinimumSize(QtCore.QSize(300, 40))
        self.btnOkay.setObjectName(_fromUtf8("btnOkay"))
        self.horizontalLayout.addWidget(self.btnOkay)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        spacerItem1 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem1)
        InfoWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(InfoWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        InfoWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(InfoWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        InfoWindow.setStatusBar(self.statusbar)

        self.retranslateUi(InfoWindow)
        QtCore.QMetaObject.connectSlotsByName(InfoWindow)

    def retranslateUi(self, InfoWindow):
        InfoWindow.setWindowTitle(_translate("InfoWindow", "Leap Theremin Experiment", None))
        self.lblInfo.setText(_translate("InfoWindow", "THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - THIS EXPERIMENT IS VERY BLAH BLAH AND SO NAGNAGNAG - ", None))
        self.btnOkay.setText(_translate("InfoWindow", "Okay", None))

