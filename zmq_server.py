import zmq
from time import sleep
context = zmq.Context()

socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
while True:
    a = socket.recv()
    print a
    sleep(1)
    socket.send("World")