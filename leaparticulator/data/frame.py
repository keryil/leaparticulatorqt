'''
Created on Mar 6, 2014

@author: kerem
'''

from random import randint, random

class InteractionBox(object):
    center = None
    width = None
    height = None
    depth = None
    is_valid = None
    def __init__(self):
        pass

    def set(self, box):
        self.center = box.center.to_tuple()
        self.width = box.width
        self.height = box.height
        self.depth = box.depth
        self.is_valid = box.is_valid
        return self

class LeapHand(object):
    id = None
    frame = None
    palm_position = None
    stabilized_palm_position = None
    palm_velocity = None
    palm_normal = None
    direction = None
    sphere_center = None
    sphere_radius = None
    time_visible = None
    is_valid = None
    _translation = None
    _translation_prob = None

    def __init__(self, hand, frame, random=False):
        if random:
            self.id = 1
            self.frame = None
            self.stabilized_palm_position = (randint(0,100),randint(0,100),randint(0,100))
            self.palm_normal = (randint(0,100),randint(0,100),randint(0,100))
            self.palm_position = (randint(0,100),randint(0,100),randint(0,100))
            self.palm_velocity = (randint(0,100),randint(0,100),randint(0,100))
            self.direction = (randint(0,100),randint(0,100),randint(0,100))
            self.sphere_center = (randint(0,100),randint(0,100),randint(0,100))
            self.time_visible = randint(0,100)
            self.is_valid = True
        else:
            self.id = hand.id
            self.frame = frame
            self.stabilized_palm_position = hand.stabilized_palm_position.to_tuple()
            self.palm_normal = hand.palm_normal.to_tuple()
            self.palm_position = hand.palm_position.to_tuple()
            self.palm_velocity = hand.palm_velocity.to_tuple()
            self.direction = hand.direction.to_tuple()
            self.sphere_center = hand.sphere_center.to_tuple()
            self.time_visible = hand.time_visible
            self.is_valid = hand.is_valid

class LeapFrame(object):
    '''
    This is a pure python clone of the Leap Motion Controller
    frame objects. It is written to be picklable, unlike the
    original, SWIG-generated frame objects. It does not include
    anything finer-grained than hand movements i.e. no pointables,
    fingers or tools.
    '''
    id = None
    timestamp = None
    hands = None
    interaction_box = None
    current_frames_per_second = None
    is_valid = None


    def __init__(self, frame, random=False):
        '''
        Constructs a new python frame from the original frame
        '''
        if random:
            self.id = randint(0,100)
            self.timestamp = randint(0,10000)
            self.hands = [LeapHand(None, None, random=random)]
            self.interaction_box = InteractionBox()
            self.interaction_box.center = (randint(0,100),randint(0,100),randint(0,100))
            self.interaction_box.width = randint(0,100)
            self.interaction_box.height = randint(0,100)
            self.interaction_box.depth = randint(0,100)
            self.current_frames_per_second = randint(0,100)
            self.is_valid = True

        else:
            self.id = frame.id
            self.timestamp = frame.timestamp
            self.hands = [LeapHand(hand, self) for hand in frame.hands]
            self.interaction_box = InteractionBox().set(frame.interaction_box)
            self.current_frames_per_second = frame.current_frames_per_second
            self.is_valid = frame.is_valid

    def get_stabilized_position(self):
        """
        Shortcut to getting the stabilized position of 
        the first available hand.
        """
        return self.hands[0].stabilized_palm_position

    def hand(self, id):
        """
        The Hand object with the specified ID in this frame.
        """
        pass

    def gestures(self, sinceFrame):
        """
        Returns a GestureList containing all gestures that have
        occured since the specified frame.
        """
        pass

    def toJSON(self):
        import jsonpickle
        return jsonpickle.encode(self)

    
def generateRandomSignal(duration):
    """
    Returns a randomly generated list of LeapFrame objects
    for testing purposes.
    """
    from jsonpickle import encode
    lst = []
    for i in range(duration):
        lst.append(encode(LeapFrame(None, random=True)))
    return lst
        