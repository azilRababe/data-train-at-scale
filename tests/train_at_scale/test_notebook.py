from nbresult import ChallengeResultTestCase

class TestNotebook(ChallengeResultTestCase):

    def test_y_pred(self):
        y_new = self.result.y_new.flat[0].tolist()
        self.assertIsInstance(y_new, float)
