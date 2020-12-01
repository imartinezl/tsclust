# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import pprint
import sys

pprint.pprint(sys.path)

from tsclust import example


class TestSimple(unittest.TestCase):
    def test_add_one(self):
        self.assertEqual(example.add_one(5), 6)


if __name__ == "__main__":
    unittest.main()
