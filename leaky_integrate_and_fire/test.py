import unittest
import LIF


class TestLIF(unittest.TestCase):

    def test_runnable_Step(self):
        """Test if LIF_Step is runnable."""
        LIF.LIF_Step()

    def test_runnable_Sinus(self):
        """Test if LIF_Step is runnable."""
        LIF.LIF_Sinus()
