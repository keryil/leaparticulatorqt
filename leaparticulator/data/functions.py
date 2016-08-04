from collections import namedtuple

import jsonpickle
import pandas as pd

# this is a list of refactored classes which need to be
# replaced with their new FQ names for jsonpickle to be
# able to unpickle them.

to_replace = [
    ("LeapFrame.", "leaparticulator.data.frame."),
    ("TestQuestion.", "leaparticulator.question."),
    ("QtUtils.Meaning", "leaparticulator.data.meaning.Meaning"),
    ("QtUtils.", "leaparticulator.question."),
    ("leaparticulator.question.FeaturelessMeaning", "leaparticulator.data.meaning.FeaturelessMeaning")
]


def refactor_old_references(string):
    for stuff in to_replace:
        string = string.replace(*stuff)
    return string


def recursive_decode(lst, verbose=False):
    """
    Recursively decode a log file line encoded by JSON into a list of Python objects.
    :param lst:
    :param verbose:
    :return:
    """
    if verbose:
        print "Decoding %s" % (str(lst)[:100])

    if isinstance(lst, str) or isinstance(lst, unicode):
        try:
            # the arg lst may or may not be a pickled obj itself
            # if not isinstance(lst, str):
            #     raise TypeError
            while isinstance(lst, str) or isinstance(lst, unicode):
                lst = jsonpickle.decode(lst)
        except TypeError, err:
            print err
            pass
        except KeyError, err:
            print "Error: %s" % err
            print "String: %s" % lst
            raise err
    if isinstance(lst, dict):
        if "py/object" in lst.keys():
            if verbose:
                print "Unpickle obj (type: %s)..." % lst['py/object']
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
    try:
        assert "py/object" not in str(lst)
    except Exception, ex:
        print ex, str(lst)
        raise Exception(type(lst), str(lst), ex)
    return lst


# converts a list of objects to a list of their
# string representations
toStr = lambda x: map(str, x)


def fromFile(filename, no_practice=False):
    """
    Read experiment log from file.
    :param filename:
    :param no_practice:
    :return:
    """
    from os import path
    if str.startswith(path.split(filename)[-1], "P2P"):
        return fromFile_p2p(filename)
    lines = open(filename).readlines()
    lines = [refactor_old_references(line) for line in lines]
    images = jsonpickle.decode(lines[0])
    responses = recursive_decode(lines[1])
    test_results = jsonpickle.decode(lines[2])
    test_results = _expandTestResults(test_results, images)
    responses = _expandResponsesNew(responses, images)

    if not no_practice:
        responses_practice = recursive_decode(lines[3])
        test_results_practice = jsonpickle.decode(lines[4])
        test_results_practice = _expandTestResults(test_results_practice, images)
        responses_practice = _expandResponsesNew(responses_practice, images)
        return responses, test_results, responses_practice, test_results_practice, images

    return responses, test_results, None, None, images


def new_recursive_decode(string, verbose=False):
    def print_(stuff):
        if verbose:
            print stuff

    obj = string
    if (isinstance(string, str) or isinstance(string, unicode)) \
            and "py/object" in string:
        print_("Decoding string: {}".format(string))
        obj = jsonpickle.decode(string)

    if isinstance(obj, dict):
        if ("py/object" in obj):
            print_("Detected a dict with py/object as key: {}".format(obj))
            new_obj = {}
            for k, v in obj.items():
                try:
                    new_obj[k] = jsonpickle.decode(v)
                except:
                    new_obj[k] = new_recursive_decode(v)

    if isinstance(obj, list):
        obj = [new_recursive_decode(o) for o in obj]
    if verbose:
        print_("Returning object: {}".format(obj))
    return obj


def fromFile_p2p(filename):
    """
    Read experimental data from a P2P log file.
    :param filename:
    :return:
    """
    def client_hook(obj, context):
        context['participants'] = obj
        context['responses'] = {p: {} for p in obj}
        return context

    def meaning_hook(obj, context):
        context['meanings'] = obj
        return context

    def round_hook(obj, nround, phase, context):
        round_summary = obj
        speaker = round_summary.speaker
        hearer = round_summary.hearer
        signal = round_summary.signal
        image_pointer = round_summary.image_pointer
        meanings = context['meanings']
        responses = context['responses']

        # if this fails, there is something seriously
        # wrong about this log file.
        assert round_summary.image in meanings

        try:
            resp = responses[speaker][phase]
        except KeyError:
            responses[speaker][phase] = {str(meaning): None for meaning in meanings[:image_pointer]}
            resp = responses[speaker][phase]

        # we only want the successful rounds
        if round_summary.success:
            # print resp.keys()
            resp[str(round_summary.image)] = signal
            # print round_summary.image_pointer
            # responses.append(round_summary)
        return context

    context_dict = dict(responses={},
                        meanings=None)

    context_dict = process_p2p_log(filename,
                                   clients_hooks=[client_hook],
                                   meanings_hooks=[meaning_hook],
                                   round_hooks=[round_hook],
                                   context_dict=context_dict)

    return namedtuple("RoundSummaryTuple", ["responses", "images"])(responses=context_dict['responses'],
                                                                    images=context_dict['meanings'])


def process_p2p_log(filename, clients_hooks=[], meanings_hooks=[], round_hooks=[],
                    context_dict={}, reverse=False, nphases=8):
    """
    Takes a P2P log file, and applies clients_hook to the first line,
    meanings_hook to the second, and round_hook for every following round.
    All operations use context_dict to read/write data.

    round_hooks members are of the form function(object, nround, phase, context_dict).
    Other hook members are of the form function(object, context_dict).
    All hook members are to return the context_dict.

    :param context_dict:
    :param filename:
    :param clients_hooks:
    :param meanings_hooks:
    :param round_hooks:
    :return:
    """
    decode = jsonpickle.decode
    with open(filename) as f:
        phase = -1
        last_pointer = -1
        lines = zip(range(-2, 40000), f)

        if reverse:
            lines = reversed(lines)
            phase = nphases
            last_pointer = 10000

        for i, line in lines:
            obj = decode(line.replace("__main__", "leaparticulator.p2p.server"))
            if i >= 0:
                phase_change = ((obj.image_pointer > last_pointer) and not reverse) \
                               or ((obj.image_pointer < last_pointer) and reverse)
                obj.signal = map(decode, obj.signal)

                if phase_change:
                    phase += 1
                    if reverse:
                        phase -= 2

                    last_pointer = obj.image_pointer
                    print "Phase {} {} at round {}".format(phase, "ends" if reverse else "starts", i)

                for round_hook in round_hooks:
                    context_dict = round_hook(obj, i, phase, context_dict)
            elif i == -2:
                for clients_hook in clients_hooks:
                    context_dict = clients_hook(obj, context_dict)
            elif i == -1:
                for meanings_hook in meanings_hooks:
                    context_dict = meanings_hook(obj, context_dict)

    return context_dict


def toPandas_p2p(filename, nphases=8):
    """
    Takes a P2P log file, and returns TWO pandas.DataFrame objects, one for
    response data, one for question data, respectively. For the response data,
    the "client" is the speaker. For the test data, the "client" is the hearer.

    nphases is optional (and deprecated). It informs the algorithm of the number
    of phases in the experimental run. It used to be needed for runs that don't
    have exactly 15 meanings.
    :param nphases:
    :param filename:
    :return:
    """
    from leaparticulator.constants import palmToAmpAndFreq, palmToAmpAndMel
    columns_response = ['round', 'client', 'phase', 'image', 'data_index', 'x', 'y', 'z', 'frequency', 'mel',
                        'amplitude']
    columns_test = ['client', 'phase', 'image0', 'image1', 'image2', 'image3', 'answer', 'given_answer', 'success']

    def response_hook(obj, nround, phase, context):
        client = obj.speaker
        meaning = obj.image
        success = obj.success
        assert success == (obj.image == obj.guess)
        signal = obj.signal
        # guarantee that we only capture the last success
        if signal and success and \
                ((client, phase, str(meaning)) not in context['phase_and_meaning']):
            context['phase_and_meaning'][(client, phase, str(meaning))] = True
            for i, frame in reversed(list(enumerate(signal))):
                x, y, z = frame.get_stabilized_position()
                amplitude, hertz = palmToAmpAndFreq((x, y))[1]
                mel = palmToAmpAndMel((x, y))[1]
                context['lst_responses'].append(
                    pd.Series([nround, client, phase, meaning, i, x, y, z, hertz, mel, amplitude],
                              index=columns_response))
        return context

    def question_hook(obj, nround, phase, context):
        rnd = obj

        # the owner of the test is the hearer
        client = rnd.hearer
        answer = rnd.image
        given_answer = rnd.guess
        success = answer == given_answer
        image0 = image1 = image2 = image3 = None
        try:
            image0 = rnd.options[0]
            image1 = rnd.options[1]
            image2 = rnd.options[2]
            image3 = rnd.options[3]
        except IndexError:
            pass

        context['lst_questions'].append(
            pd.Series([client, phase, image0, image1, image2, image3, answer, given_answer, success],
                      index=columns_test))
        return context

    context_dict = dict(lst_questions=[],
                        lst_responses=[],
                        phase_and_meaning={})

    context_dict = process_p2p_log(filename,
                                   round_hooks=[question_hook,
                                                response_hook],
                                   context_dict=context_dict,
                                   reverse=True,
                                   nphases=nphases)
    reverse_list = lambda x: list(reversed(x))

    # calculate the phase offset, and apply it if nonzero
    min_phase = context_dict['lst_questions'][-1]['phase']
    if min_phase:
        print "Correcting for {} phases (offset: {})...".format(nphases - min_phase, - min_phase)
        for lst in (context_dict['lst_questions'], context_dict['lst_responses']):
            for row in lst:
                row['phase'] -= min_phase

    df_test = pd.DataFrame(reverse_list(context_dict['lst_questions']), columns=columns_test)
    df_response = pd.DataFrame(reverse_list(context_dict['lst_responses']), columns=columns_response)
    # print context_dict
    return df_response, df_test

def fromFile_old(filename):
    """
    Read experimental data from an old file. Only here for backward compatibility.
    :param filename:
    :return:
    """
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
    """
    Updates the responses dictionary to include the actual meaning objects instead of their indexes.
    :param responses:
    :param images:
    :return:
    """
    d = {}
    for client in responses:
        d[client] = {}
        for phase in responses[client]:
            d[client][phase] = {}
            for image in responses[client][phase]:
                d[client][phase][images[int(phase)][int(image)]] = responses[client][phase][image]
    return d


def _expandResponses(responses, images):
    for client in responses:
        for phase in responses[client]:
            d = {}
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
    """
    Extends the test results by replacing meaning indexes with the actual meaning objects.
    :param results:
    :param images:
    :return:
    """
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
    if data is None:
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
    from leaparticulator.notebooks.HandTrajectory2SignalTrajectory import calculate_amp_and_freq

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
            # print "Phase:", phase
            # print "Image indexes:", responses[client][phase].keys()
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
            # print "Phase:", phase
            d[phase] = []
            for index, question in enumerate(test_results[client][phase]):
                # print "Question:", index, question
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

def logToPandasFrame(logfile):
    """
    Returns the given log file as a pandas frame, structured identically to the CSV.
    :param logfile:
    :return:
    """
    from itertools import product
    import pandas as pd

    results = fromFile(logfile)

    responses, test_results, responses_practice, test_results_practice, images = results
    responses = responses['127.0.0.1']
    responses_p = responses_practice['127.0.0.1']
    phases = map(str, range(3))
    meanings = responses[phases[0]].keys()
    columns = ['phase', 'meaning', 'frame_index', 'x', 'y', 'practice']
    all_data = pd.DataFrame(columns=columns)
    #     print all_data
    grand_index = 0
    for phase, meaning in product(phases, meanings):
        for response_dict in (responses, responses_p):
            frame = pd.DataFrame(columns=columns)
            if meaning not in response_dict[phase]:
                continue
            traj = []
            for leapframe in response_dict[phase][meaning]:
                traj.append(leapframe.get_stabilized_position()[:2])
            # traj = [f.get_stabilized_position()[:2] if meaning in response_dict[phase] for f in response_dict[phase][meaning]]
            # if response_dict == responses:
            #     xy_lists.append(traj)
            # else:
            #     xy_lists_p.append(traj)
            index0 = grand_index
            index_f = grand_index + len(traj)
            index = range(grand_index, index_f)
            grand_index += len(traj)

            phase_l = []
            meaning_l = []
            xs = []
            ys = []
            i_s = []
            for i, (x, y) in enumerate(traj):
                xs.append(x)
                ys.append(y)
                i_s.append(i)
                phase_l.append(phase)
                meaning_l.append(meaning)
            practice = [response_dict == responses_p for _ in xs]
            for field, lst in zip(columns, [phase_l, meaning_l, i_s, xs, ys, practice]):
                frame[field] = pd.Series(name=field, data=lst, index=index)
            all_data = all_data.append(frame)
    return all_data


# if __name__ == "__main__":
# print(fromFile("./logs/Heikki Pilot.exp.log"))







import os
import sys
from contextlib import contextmanager


@contextmanager
def stderr_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    original_stderr = sys.stderr
    fd = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as f:
            _redirect_stderr(to=f)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stderr(to=old_stderr)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
