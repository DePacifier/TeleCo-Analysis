import unittest
import pandas as pd
import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.data_cleaner import DataCleaner

timeval = datetime.datetime.now()
data = {'float_values': [5.28, 2.48, 5.28], 'int_values': [
    1, 2, 1], 'str_values': ['abebe', 'debebe', 'abebe'], 'dtvalue': [timeval, timeval, timeval]}
df = pd.DataFrame(data)

class TestCases(unittest.TestCase):
    def test_class_creation(self):
        data_cleaner = DataCleaner(df)
        self.assertEqual(df.info(), data_cleaner.df.info())

    def test_remove_unwanted_columns(self):
        """
        Test that it retunrs the average of a given list
        """
        data_cleaner = DataCleaner(df)
        rm_col_name = 'float_values'
        data_cleaner.remove_unwanted_columns([rm_col_name])
        self.assertTrue(rm_col_name not in data_cleaner.df.columns.tolist())

    def test_separate_date_time_column(self):
        """
        Provide an assertion level for arg input
        """
        data_cleaner = DataCleaner(df)
        data_cleaner.separate_date_time_column('dtvalue', 'newdt')
        self.assertTrue('newdt_date' in data_cleaner.df.columns.tolist(
        ) and 'newdt_time' in data_cleaner.df.columns.tolist())

    def test_change_columns_type_to(self):
        """
        Provide an assertion level for arg input
        """
        data_cleaner = DataCleaner(df)
        data_cleaner.change_columns_type_to(['str_values'], '|S10')
        data_type = data_cleaner.df.dtypes['str_values']
        self.assertEqual(data_type,r'|S10')

    def test_remove_duplicates(self):
        data_cleaner = DataCleaner(df)
        data_cleaner.remove_duplicates()
        self.assertEqual(data_cleaner.df.shape[0], df.shape[0])

    def test_save_clean_data(self):
        data_cleaner = DataCleaner(df)
        data_cleaner.save_clean_data('test.csv')
        self.assertTrue('test.csv' in os.listdir())
        os.remove('test.csv')


if __name__ == '__main__':
    unittest.main()
