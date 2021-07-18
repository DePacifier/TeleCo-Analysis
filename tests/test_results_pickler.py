import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.results_pickler import ResultPickler

class TestCases(unittest.TestCase):
    def test_add_data(self):
        """
        Test that it retunrs the average of a given list
        """
        data = [1, 2, 3]
        pickler = ResultPickler()
        pickler.add_data('new_value', data)
        result = pickler.data['new_value']
        self.assertEqual(result, data)

    def test_save_data(self):
        """
        Provide an assertion level for arg input
        """
        data = [1, 2, 3]
        pickler = ResultPickler()
        pickler.add_data('new_value', data)
        pickler.save_data('test.pickle')
        self.assertTrue('test.pickle' in  os.listdir())

    def test_load_data(self):
        """
        Provide an assertion level for arg input
        """
        data = [1, 2, 3]
        pickler = ResultPickler()
        pickler.load_data('test.pickle')
        result = pickler.data['new_value']
        self.assertEqual(result, data)

    def test_get_data(self):
        """
        Provide an assertion level for arg input
        """
        data = [1, 2, 3]
        pickler = ResultPickler()
        pickler.add_data('new_value', data)
        result = pickler.get_data()
        self.assertEqual(pickler.data, result)


if __name__ == '__main__':
    unittest.main()
