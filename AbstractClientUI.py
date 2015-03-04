class AbstractClientUI(object):
    
    def go_test(self):
        """
        Virtual method to initiate the next test phase.
        """
        raise NotImplementedError()

    def next_phase(self, practice):
        """
        Virtual method that either steps the client through the first 
        window or takes them to the next phase's pre-phase 
        screen.
        """
        raise NotImplementedError()
    
    def on_new_test_question(self, question):
        """
        Callback for when a new test question is received
        from the server. 
        """
        raise NotImplementedError()

    def on_new_picture(self, data):
        """
        Callback for when a new picture is received from 
        the server.
        """
        raise NotImplementedError()
    
    def extend_last_signal(self, pickled_data):
        """
        Callback for the arrival of a new Leap frame while 
        the theremin is recording.
        """
        raise NotImplementedError()
    
    def exit(self):
        raise NotImplementedError()