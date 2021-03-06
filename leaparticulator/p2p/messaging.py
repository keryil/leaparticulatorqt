from leaparticulator import constants


class RoundData(object):
    def __init__(self, isSpeaker, image):
        assert isSpeaker in (True, False)
        self.isSpeaker = isSpeaker
        self.image = image


class ResponseData(object):
    def __init__(self, signal, image, options=None):
        self.signal = signal
        self.image = image
        self.options = options


# RoundData = namedtuple("RoundData", ["isSpeaker", "image"])
# ResponseData = namedtuple("ResponseData", ["signal", "image"])

class LeapP2PMessage(object):
    """
    This class encapsulates all messages passed in the framework. Its 
    main function is to couple the control instructions (as listed in 
    the module Constants) with their associated data for easy, one-step
    transmission. 
    """

    def __init__(self):
        self.data = None
        self.instruction = None

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)


class FeedbackMessage(LeapP2PMessage):
    """
    Signals the success/failure of the round
    """
    instruction = constants.FEEDBACK

    def __init__(self, target_image, chosen_image, success, image_pointer):
        self.success = success
        self.target_image = target_image
        self.chosen_image = chosen_image
        self.data = {"success": success, "target": target_image, "chosen": chosen_image}
        self.image_pointer = image_pointer


class InitMessage(LeapP2PMessage):
    """
    Informs the server of client parameters.
    """
    client_id = None
    instruction = constants.INIT

    def __init__(self, client_id):
        self.client_id = client_id


class StartMessage(LeapP2PMessage):
    """
    Starts the session.
    """

    def __init__(self):
        self.instruction = constants.START


class ImageListMessage(LeapP2PMessage):
    """
    Contains a list of images to be used in the session
    """

    def __init__(self, images):
        self.instruction = constants.IMAGE_LIST
        self.data = images
        assert self.data is not None
        assert len(self.data) != 0
        # assert len(self.data[0]) != 0


class StartRoundMessage(LeapP2PMessage):
    """
    Contains whether or not the receiver is a speaker, and also 
    the topic image.
    """

    def __init__(self, isSpeaker, image):
        self.data = RoundData(isSpeaker, image)
        self.instruction = constants.START_ROUND


class EndRoundMessage(LeapP2PMessage):
    """
    Signals the end of round. Equivalent to END_QUESTION in single 
    user mode.
    """

    def __init__(self):
        self.instruction = constants.END_ROUND


class ResponseMessage(LeapP2PMessage):
    """
    Represents responses to images or signals. In the case of signal creation, 
    the speaker sends an instance of this message to the server, and puts 
    their own signal into the signal field. In the case of signal recognition, 
    the hearer receives an instance of this message from the server,
    replays the signal, and puts its prediction of the topic object into the 
    image field to send it back to the server.
    """

    def __init__(self, signal, image, options=None):
        self.instruction = constants.RESPONSE
        self.data = ResponseData(signal=signal, image=image, options=options)


class EndSessionMessage(LeapP2PMessage):
    """
    Sent to clients to indicate the experimental session is over.
    """

    def __init__(self):
        self.instruction = constants.END_SESSION
