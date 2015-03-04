from collections import namedtuple
import Constants

RoundData = namedtuple("RoundData", ["isSpeaker", "image"]) 
ResponseData = namedtuple("ResponseData", ["signal", "image"])

class LeapP2PMessage(object):
    """
    This class encapsulates all messages passed in the framework. Its 
    main function is to couple the control instructions (as listed in 
    the module Constants) with their associated data for easy, one-step
    transmission. 
    """
    data = None
    instruction = None
    def __init__(self):
        pass

class FeedbackMessage(LeapP2PMessage):
    """
    Signals the success/failure of the round
    """
    instruction = Constants.FEEDBACK
    def __init__(self, target_image, chosen_image, success):
        self.success = success
        self.target_image = target_image
        self.chosen_image = chosen_image

class InitMessage(LeapP2PMessage):
    """
    Informs the server of client parameters.
    """
    client_id = None
    instruction = Constants.INIT
    def __init__(self, client_id):
        self.client_id = client_id

class StartMessage(LeapP2PMessage):
    """
    Starts the session.
    """
    def __init__(self):
        self.instruction = Constants.START

class ImageListMessage(LeapP2PMessage):
    """
    Contains a list of images to be used in the session
    """
    def __init__(self, images):
        self.instruction = Constants.IMAGE_LIST
        self.data = images
        assert self.data is not None
        assert len(self.data) != 0

class StartRoundMessage(LeapP2PMessage):
    """
    Contains whether or not the receiver is a speaker, and also 
    the topic image.
    """
    def __init__(self, isSpeaker, image):
        self.data = RoundData(isSpeaker, image)
        self.instruction = Constants.START_ROUND

class EndRoundMessage(LeapP2PMessage):
    """
    Signals the end of round. Equivalent to END_QUESTION in single 
    user mode.
    """
    def __init__(self):
        self.instruction = Constants.END_ROUND

class ResponseMessage(LeapP2PMessage):
    """
    Represents responses to images or signals. In the case of signal creation, 
    the speaker sends an instance of this message to the server, and puts 
    their own signal into the signal field. In the case of signal recognition, 
    the hearer receives an instance of this message from the server, 
    replays the signal, and puts its prediction of the topic object into the 
    image field to send it back to the server.
    """
    def __init__(self, signal, image):
        self.instruction = Constants.RESPONSE
        self.data = ResponseData(signal=signal, image=image)