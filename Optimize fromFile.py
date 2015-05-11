# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

% load_ext
line_profiler
% lprun?

# <codecell>

from leaparticulator.data.functions import fromFile, fromFile_old, _expandResponsesNew

# <codecell>

f = "logs/13202126514.2.exp.log"
% lprun - f
_expandResponsesNew
fromFile(f)

# <codecell>

import jsonpickle


def recursive_decode(lst, verbose=False):
    if verbose:
        print "Decoding %s" % (str(lst)[:100])
    try:
        # the arg lst may or may not be a pickled obj itself
        lst = jsonpickle.decode(lst)
    except TypeError:
        pass
    if isinstance(lst, dict):
        if "py/objects" in lst.keys():
            if verbose:
                print "Unpickle obj..."
            lst = jsonpickle.decode(lst)
        else:
            if verbose:
                print "Unwind dict..."
            lst = {i: recursive_decode(lst[i]) for i in lst.keys()}
    elif isinstance(lst, list):
        if verbose:
            print "Decode list..."
        lst = [recursive_decode(l) for l in lst]
    else:
        if verbose:
            print "Probably hit tail..."
    return lst


lines = open(f).readlines()
images = jsonpickle.decode(lines[0])


def old_way(lines):
    responses = jsonpickle.decode(lines[1])
    responses = _expandResponses(responses, images)


def new_way(lines):
    responses = recursive_decode(lines[1])
    responses = {client: {
    phase: {images[int(phase)][int(image)]: responses[client][phase][image] for image in responses[client][phase]} for
    phase in responses[client]} for client in responses}

# for client in responses:
#         for phase in responses[client]:
#             d = responses[client][phase]
#             responses[client][phase] = {images[int(phase)][int(image)]:d[image] for image in d}


# print "***********"
% timeit - n
50
old_way(lines)
% timeit - n
50
new_way(lines)
# l = responses["127.0.0.1"]["1"]["1"]
# [jsonpickle.decode(r) for r in l]

# <codecell>

% timeit
fromFile(f)
% timeit
fromFile_old(f)

# <codecell>


