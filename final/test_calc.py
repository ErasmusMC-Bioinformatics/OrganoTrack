import unittest
import calc

# Test cases

# Class to test calc.py, inherited from unittest.TestCase
class TestCalc(unittest.TestCase):
    def test_add(self):
        result = calc.add(10, 5)
        self.assertEqual(result, 15)