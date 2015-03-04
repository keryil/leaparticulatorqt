
import sys
from PyQt4 import QtGui, QtCore
from QtUtils import setButtonIcon

def main():
    
    app = QtGui.QApplication(sys.argv)

    path1 = "/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/img/meanings/newexperiment/333.png"
    path2 = "/shared/Dropbox/ABACUS/Workspace/LeapArticulatorQt/img/false.png"

    px0 = QtGui.QPixmap(250,250)
    px0.fill(QtCore.Qt.transparent)

    px1 = QtGui.QPixmap(path1)
    
    px2 = QtGui.QPixmap(path2)


    painter = QtGui.QPainter(px0)
    # painter.begin(px0)
    painter.drawPixmap(0,0,px1)
    painter.drawPixmap(0,0,px2)
    painter.end()

    w = QtGui.QWidget()
    w.resize(250, 250)
    w.move(300, 300)
    w.setWindowTitle('Simple')

    btn = QtGui.QPushButton(parent=w)
    setButtonIcon(btn, px0)

    w.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
