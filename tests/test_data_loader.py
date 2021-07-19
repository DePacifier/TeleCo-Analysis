import unittest
from unittest import result
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.data_loader import load_df_from_csv, load_df_from_excel, optimize_df

data = {'float_values': [5.28, 2.48], 'int_values': [1, 2], 'str_values': ['abebe','debebe']}
df = pd.DataFrame(data)

class TestCases(unittest.TestCase):
    def test_test_data_availability(self):
        df.to_csv('test.csv')
        df.to_excel('test.xlsx', engine='openpyxl')
        self.assertTrue('test.csv' in os.listdir() and 'test.xlsx' in os.listdir())

    def test_load_df_from_csv(self):
        """
        Test that it retunrs the average of a given list
        """
        df.to_csv('test.csv')
        result = load_df_from_csv('test.csv')
        data = optimize_df(df)
        self.assertEqual(result.info(), data.info())

    def test_load_df_from_excel(self):
        """
        Provide an assertion level for arg input
        """
        df.to_excel('test.xlsx', engine='openpyxl')
        result = load_df_from_excel('test.xlsx')
        data = optimize_df(df)
        self.assertEqual(result.info(), data.info())

    def test_optimize_df(self):
        """
        Provide an assertion level for arg input
        """
        result = optimize_df(df)
        df['float_values'] = df['float_values'].astype('float32')
        df['int_values'] = df['int_values'].astype('uint8')
        self.assertEqual(result.info(), df.info())

    def test_test_data_removed(self):
        os.remove('test.csv')
        os.remove('test.xlsx')
        self.assertTrue('test.csv' not in os.listdir() and 'test.xlsx' not in os.listdir())


if __name__ == '__main__':
    unittest.main()
