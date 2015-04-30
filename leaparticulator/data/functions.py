import jsonpickle
import pandas as pd

from leaparticulator.data.hmm import HMM, reconstruct_hmm, reduce_hmm


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

# converts a list of objects to a list of their
# string representations
toStr = lambda x: map(str, x)


def fromFile(filename):
    lines = open(filename).readlines()
    images = jsonpickle.decode(lines[0])
    responses = recursive_decode(lines[1])
    test_results = jsonpickle.decode(lines[2])
    responses_practice = recursive_decode(lines[3])
    test_results_practice = jsonpickle.decode(lines[4])
    # return _expandResponses(responses, images), _expandTestResults(test_results, images)

    # print images
    # tests, res = convertToPandas(images, responses, test_results)
    test_results = _expandTestResults(test_results, images)
    test_results_practice = _expandTestResults(test_results_practice, images)
    responses_practice = _expandResponsesNew(responses_practice, images)
    responses = _expandResponsesNew(responses, images)
    return responses, test_results, responses_practice, test_results_practice, images


def fromFile_old(filename):
    lines = open(filename).readlines()
    images = jsonpickle.decode(lines[0])
    responses = jsonpickle.decode(lines[1])
    test_results = jsonpickle.decode(lines[2])
    responses_practice = jsonpickle.decode(lines[3])
    test_results_practice = jsonpickle.decode(lines[4])
    # return _expandResponses(responses, images), _expandTestResults(test_results, images)

    # print images
    # tests, res = convertToPandas(images, responses, test_results)
    test_results = _expandTestResults(test_results, images)
    test_results_practice = _expandTestResults(test_results_practice, images)
    responses_practice = _expandResponsesNew(responses_practice, images)
    responses = _expandResponsesNew(responses, images)
    return responses, test_results, responses_practice, test_results_practice, images


def _expandResponsesNew(responses, images):
    return {client: {phase: {images[int(phase)][int(image)]: responses[client][phase][image] \
                             for image in responses[client][phase]} \
                     for phase in responses[client]} \
            for client in responses}


def _expandResponses(responses, images):
    for client in responses:
        for phase in responses[client]:
            d = {}
            firstTimeFrame = None
            lastTimeFrame = None
            for image in responses[client][phase]:
                # print phase, imaged
                im = images[int(phase)][int(image)]
                # print responses[client][phase][image][0]
                frames = responses[client][phase][image]
                # print frames[0]
                d[im] = [jsonpickle.decode(frame) for frame in frames]
                lastTimeFrame = d[im][-1].timestamp
                firstTimeFrame = d[im][0].timestamp
                dif = float(lastTimeFrame - firstTimeFrame)
                for frame in d[im]:
                    # normalize time to lie between 0 and 1
                    try:
                        frame.timestamp = (frame.timestamp - firstTimeFrame) / dif
                    except ZeroDivisionError, e:
                        print e
                        print (("Participant response at phase %s image %s (%s) consists of " +
                                "a single timeframe. This will be handled appropriately, " +
                                "and the parse will not fail. But if " +
                                "you are reading this while parsing a file that is not " +
                                "logs/123R0126514.1r.exp.log, pm Kerem.") % (phase,
                                                                             image,
                                                                             im))

                        frame.timestamp = 0
            responses[client][phase] = d
    return responses


def _expandTestResults(results, images):
    for client in results:
        for phase in results[client]:
            for question in results[client][phase]:
                question.pics = [images[int(phase)][pic] for pic in question.pics]
                question.answer = images[int(phase)][int(question.answer)]
                question.given_answer = images[int(phase)][int(question.given_answer)]
    return results


def toCSV(filename, delimiter="|", data=None):
    """
    Converts exp.log files to CSV files, one for responses, one for tests.
    The filename will be identical to the log file's name except its extension
    will be .csv, and before the extension it will indicate whether it is
    a response file or a test file.

    If the data parameter is not None, it is assumed to be a 5-tuple of (responses,
    test_results, responses_practice, test_results_practice, images). In this case,
    the log file is not read, and things to WAAAY faster, given one has already parsed
    to log file for some other purpose.

    Format for responses is [client, is_practice, phase, image, data_index, x, y, z]
    Format for tests is [client, is_practice, phase, images, answer, given_answer]
    """
    responses = test_results = responses_practice = test_results_practice = images = None
    if data == None:
        data = fromFile(filename)
    responses, test_results, responses_practice, test_results_practice, images = data

    csv_filename = filename.replace(".log", ".responses.csv")

    print("Saving responses into %s" % csv_filename)
    doublequote = lambda x: "\"%s\"" % str(x)
    # first, the responses
    with open(csv_filename, "w") as csv:
        # write headers
        csv.write(delimiter.join(["client", "is_practice", "phase", "image", "data_index", "x", "y", "z"]))
        csv.write("\n")
        for client in responses:
            for phase in responses[client]:
                for image in responses[client][phase]:
                    for i, frame in enumerate(responses[client][phase][image]):
                        x, y, z = frame.get_stabilized_position()
                        csv.write(
                            delimiter.join((client, '0', phase, doublequote(image), str(i), str(x), str(y), str(z))))
                        csv.write("\n")

        for client in responses_practice:
            for phase in responses[client]:
                for image in responses[client][phase]:
                    for i, frame in enumerate(responses[client][phase][image]):
                        x, y, z = frame.get_stabilized_position()
                        csv.write(
                            delimiter.join((client, '1', phase, doublequote(image), str(i), str(x), str(y), str(z))))
                        csv.write("\n")

    # calculate the amplitudes and frequencies
    from HandTrajectory2SignalTrajectory import calculate_amp_and_freq

    calculate_amp_and_freq(csv_filename, delimiter=delimiter)

    csv_filename = csv_filename.replace("responses", "tests")
    print("Saving tests into %s" % csv_filename)
    # And now the tests
    with open(csv_filename, "w") as csv:
        # write headers
        csv.write(delimiter.join(
            ["client", "is_practice", "phase", "image0", "image1", "image2", "image3", "answer", "given_answer"]))
        csv.write("\n")
        for client in test_results:
            for phase in test_results[client]:
                for question in test_results[client][phase]:
                    csv.write(delimiter.join((client,
                                              '0',
                                              phase,
                                              doublequote(doublequote(delimiter).join(toStr(question.pics))),
                                              doublequote(question.answer),
                                              doublequote(question.given_answer))))
                    csv.write("\n")

        for client in test_results_practice:
            for phase in test_results_practice[client]:
                for question in test_results_practice[client][phase]:
                    csv.write(delimiter.join((client,
                                              '1',
                                              phase,
                                              doublequote(doublequote(delimiter).join(toStr(question.pics))),
                                              doublequote(question.answer),
                                              doublequote(question.given_answer))))
                    csv.write("\n")


    # Finally, the images
    csv_filename = csv_filename.replace("tests", "images")
    print("Saving images into %s" % csv_filename)
    with open(csv_filename, "w") as csv:
        # write headers
        csv.write(delimiter.join(["image_name"]))
        csv.write("\n")
        for phase in images:
            for image in phase:
                csv.write(str(image))
                csv.write("\n")
    return data


def convertToPandas(images, responses, test_results):
    # first, convert image indexes to image paths in responses
    pd_results = {}
    for client in responses:
        clientd = {}
        pd_client = pd.DataFrame(index=responses[client].keys())
        # pd_results[client] = pd_client
        # print "Client:", client
        # print "Phases:", responses[client].keys()
        for phase in responses[client]:
            phased = {}
            # clientd[phase] = phased
            #print "Phase:", phase
            #print "Image indexes:", responses[client][phase].keys()
            for index in responses[client][phase]:
                image = images[int(phase)][int(index)]
                phased[image] = responses[client][phase][index]
            # pd_phase[index] = responses[client][phase][index]
            pd_phase = pd.Series(phased)
            pd_client[phase] = pd_phase

        del responses[client]
        responses[client] = clientd
        pd_results[client] = pd_client
    pd_results = pd.DataFrame.from_dict(pd_results, orient="index")  # , index=responses.keys())

    # the same, for the test results
    dd = pd.Panel()
    for client in test_results:
        d = {}
        # print "Client:", client
        # print "Phases:", test_results[client].keys()
        for phase in test_results[client]:
            #print "Phase:", phase
            d[phase] = []
            for index, question in enumerate(test_results[client][phase]):
                #print "Question:", index, question
                question.pics = [images[int(phase)][pic] for pic in question.pics]
                question.answer = images[int(phase)][int(question.answer)]
                question.given_answer = images[int(phase)][int(question.given_answer)]
                d[phase].append(question)
            d[phase] = pd.Series(d[phase])
        dd[client] = pd.DataFrame.from_dict(d)

    # pd_tests = pd.DataFrame.from_dict(test_results, orient="index")
    # pd_results = pd.DataFrame.from_dict(responses, orient="index")
    return pd_results, dd  # pd_tests


# return test_results





# if __name__ == "__main__":
# print(fromFile("./logs/Heikki Pilot.exp.log"))







import os
import sys
from contextlib import contextmanager


@contextmanager
def stderr_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    original_stderr = sys.stderr
    fd = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stderr(to=old_stderr)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
