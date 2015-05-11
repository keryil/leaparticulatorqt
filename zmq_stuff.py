# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

% % writefile
zmq_server.py
from time import sleep

import zmq


context = zmq.Context()

socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
while True:
    a = socket.recv()
    print a
    sleep(1)
    socket.send("World")

# <codecell>

% % writefile
zmq_client.py
import zmq

context = zmq.Context()

# Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s ..." % request)
    socket.send(b"Hello")

    # Get the reply.
    message = socket.recv()
    print("Received reply %s [ %s ]" % (request, message))

# <codecell>

% run
zmq_server.py

# <codecell>


