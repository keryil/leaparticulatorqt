Leap Articulator
============================================================

Simple synth-based theremin using Leap Motion and PyAudio, put into a server-client scheme using twisted. The clients
broadcast their signals during server-controlled sessions, and are able to receive other clients' signals through the 
server. It is a plain-text TCP protocol.

LeapSDK binaries for OS X, Windows and Linux are included.

### Requirement ###

* Leap Motion device
* LeapSDK
* PyAudio
* portaudio
* twisted
* jsonpickle
* PyQT
* pandas
* qt4reactor

### Output ###
Most logging is to stdout which should be fixed. At the end of the experiment, the server dumps an %TIMEDATE%.exp.log file that consists of the image list, the learning phase data and
the test questions along with the answers, encoded in JSON using jsonpickle. Currently I am developing some pandas wrapper which will hopefully facilitate further analysis.

### Preparation ###

You should change Constants.py file on client-side to specify the server ip/port. Same goes for the port when starting a server. 

### Execution ###

1. Connect device (1.5. Check if it works via Visualizer)

2. Launch LeapServer.py [condition] [log_directory_prefix="."]

3. Launch QTClientUI.py

4. Have fun.

5. Restart BOTH for each new experimental run.