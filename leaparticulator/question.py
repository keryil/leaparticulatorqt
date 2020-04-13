from random import sample, choice, shuffle


def produce_questions(client_responses, qty=4, n_of_images=3):
    questions = []
    # print "Test uniqueness of client responses"
    test_arr_unique(client_responses)
    assert len(client_responses) >= qty
    answers = sample(client_responses, qty)
    for answer in answers:
        q = TestQuestion(
            client_responses, n_of_images=n_of_images, answer=answer)
        questions.append(q)
        assert q.answer == answer

    # for i, response in enumerate(client_responses):
    # q = TestQuestion(
    #         client_responses, n_of_images=n_of_images, answer=response)
    #     questions.append(q)
    #     assert q.answer == response
    #     if i + 1 == qty:
    #         break
    shuffle(questions)
    # print "Test uniqueness of question bundle"
    test_uniqueness(questions)
    return questions


class TestQuestion(object):
    def __init__(self, client_responses, n_of_images=3, answer=None):
        """
        Receives a dict of response[pic]=signal, and
        constructs test question. 
        """
        # n_of_images = 4
        # dict from pics to signals
        # as in response[pic] = signal
        responses = {}
        # pics = []
        # signal = []
        # answer = None
        # given_answer = None

        self.n_of_images = n_of_images
        assert len(client_responses) == len(set(client_responses))
        images = list(client_responses.keys())
        if answer is None:
            self.pics = sample(images, n_of_images)
            self.answer = choice(self.pics)
        else:
            wrong_choices = [answer, answer, answer]
            while answer in wrong_choices:
                wrong_choices = sample(images, n_of_images - 1)
            print(wrong_choices, answer)
            wrong_choices.append(answer)
            self.pics = wrong_choices
            print(self.pics)
            # extend(wrong_choices)
            # self.pics.extend(wrong_choices)
            self.answer = answer
        shuffle(self.pics)
        self.signal = client_responses[self.answer]
        self.given_answer = None
        assert self.signal is not None


def test(rounds=100):
    """
    Tests that test questions produced in batches have no repetition of
    answers.
    """
    for i in range(rounds):
        print("Round", i + 1, "of", rounds)
        for n_of_pics in range(3, 30):
            # print n_of_pics, "symbols"
            client_responses = list(range(n_of_pics))
            for n_of_answers in range(3, n_of_pics + 1):
                # print n_of_answers, "options per question"
                qs = produce_questions(
                    client_responses, qty=n_of_pics, n_of_images=n_of_answers)
                test_uniqueness(qs)


def test_uniqueness(questions):
    answers = set()
    for q in questions:
        if q.answer in answers:
            print(q.answer, "is duplicate in set", end=' ')
            print([q.answer for q in questions])
            import sys

            sys.exit(-1)
        answers.add(q.answer)


def test_arr_unique(arr):
    if len(arr) != len(set(arr)):
        print("Duplicate in", arr)
        import sys

        sys.exit(-1)


if __name__ == "__main__":
    test()
