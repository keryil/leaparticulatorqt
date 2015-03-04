import QtQuick 2.0

Rectangle {
    width: 360
    height: 360
    Text {
        width: 250
        height: 196
        anchors.centerIn: parent
        text: "This text gives you basic info about the experiment."
        anchors.verticalCenterOffset: 0
        anchors.horizontalCenterOffset: 0
    }
    MouseArea {
        z: -2
        anchors.rightMargin: 0
        anchors.bottomMargin: 0
        anchors.leftMargin: 0
        anchors.topMargin: 0
        anchors.fill: parent
        onClicked: {
            Qt.quit();
        }
    }
}

