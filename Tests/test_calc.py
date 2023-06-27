import unittest
from Tests import calc


# Test cases

# Class to test calc.py, inherited from unittest.TestCase
class TestCalc(unittest.TestCase):
    def test_add(self):  # 3 tests under 1 test. Write good tests, not as many tests as possible.
        self.assertEqual(calc.add(10, 5), 15)
        self.assertEqual(calc.add(-1, 1), 0)  # edge case: neg + pos
        self.assertEqual(calc.add(-1, -1), -2)  # edge case: neg + neg

    def test_subtract(self):  # 3 tests under 1 test. Write good tests, not as many tests as possible.
        self.assertEqual(calc.subtract(10, 5), 5)
        self.assertEqual(calc.subtract(-1, 1), -2)  # edge case: neg + pos
        self.assertEqual(calc.subtract(-1, -1), 0)  # edge case: neg + neg

    def test_multiply(self):  # 3 tests under 1 test. Write good tests, not as many tests as possible.
        self.assertEqual(calc.multiply(10, 5), 50)
        self.assertEqual(calc.multiply(-1, 1), -1)  # edge case: neg + pos
        self.assertEqual(calc.multiply(-1, -1), 1)  # edge case: neg + neg

    def test_divide(self):  # 3 tests under 1 test. Write good tests, not as many tests as possible.
        self.assertEqual(calc.divide(10, 5), 2)
        self.assertEqual(calc.divide(-1, 1), -1)  # edge case: neg + pos
        self.assertEqual(calc.divide(-1, -1), 1)  # edge case: neg + neg
        self.assertEqual(calc.divide(5, 2), 2.5)  # edge case: floor division problem

        # self.assertRaises(ValueError, calc.divide, 10, 2)  # checks if ValueError is raised
        with self.assertRaises(ValueError):  # using a context manager to test exceptions
            calc.divide(10, 0)


if __name__ == '__main__':
    unittest.main()  # will run all tests