import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        """
        This function takes the dataframe and the column which has the bytes values
        returns the megabytesof that value

        Args:
        -----
        df: dataframe
        bytes_data: column with bytes values

        Returns:
        --------
        A series
        """
        self.df.drop(columns, axis=1, inplace=True)
        return self.df

    def separate_date_time_column(self, column: str, col_prefix_name: str) -> pd.DataFrame:
        try:

            self.df[f'{col_prefix_name}_date'] = pd.to_datetime(
                self.df[column]).dt.date
            self.df[f'{col_prefix_name}_time'] = pd.to_datetime(
                self.df[column]).dt.time

            return self.df

        except:
            print("Failed to separate the date-time column")

    def change_columns_type_to(self, cols: list, data_type: str) -> pd.DataFrame:
        try:
            for col in cols:
                self.df[col] = self.df[col].astype(data_type)
        except:
            print('Failed to change columns type')

        return self.df

    def remove_single_value_columns(self, unique_value_counts: pd.DataFrame) -> pd.DataFrame:
        drop_cols = list(
            unique_value_counts.loc[unique_value_counts['Unique Value Count'] == 1].index)
        return self.df.drop(drop_cols, axis=1, inplace=True)

    def remove_duplicates(self) -> pd.DataFrame:
        removables = self.df[self.df.duplicated()].index
        return self.df.drop(index=removables, inplace=True)

    def fill_numeric_values(self, missing_cols: list, acceptable_skewness: float = 5.0) -> pd.DataFrame:
        df_skew_data = self.df[missing_cols]
        df_skew = df_skew_data.skew(axis=0, skipna=True)
        for i in df_skew.index:
            if(df_skew[i] < acceptable_skewness and df_skew[i] > (acceptable_skewness * -1)):
                value = self.df[i].mean()
                self.df[i].fillna(value, inplace=True)
            else:
                value = self.df[i].median()
                self.df[i].fillna(value, inplace=True)

        return self.df

    def fill_non_numeric_values(self, missing_cols: list, ffill: bool = True, bfill: bool = False) -> pd.DataFrame:
        if(ffill == True and bfill == True):
            self.df[missing_cols].fillna(method='ffill', inplace=True)
            self.df[missing_cols].fillna(method='bfill', inplace=True)

        elif(ffill == True and bfill == False):
            self.df[missing_cols].fillna(method='ffill', inplace=True)

        elif(ffill == False and bfill == True):
            self.df[missing_cols].fillna(method='bfill', inplace=True)

        else:
            self.df[missing_cols].fillna(method='bfill', inplace=True)
            self.df[missing_cols].fillna(method='ffill', inplace=True)

        return self.df

    def create_new_columns_from(self, new_col_name: str, col1: str, col2: str, func) -> pd.DataFrame:
        try:
            self.df[new_col_name] = func(self.df[col1], self.df[col2])
        except:
            print("failed to create new column with the specified function")

        return self.df

    def convert_bytes_to_megabytes(self, columns: list) -> pd.DataFrame:
        """
            This function takes the dataframe and the column which has the bytes values
            returns the megabytesof that value

            Args:
            -----
            df: dataframe
            bytes_data: column with bytes values

            Returns:
            --------
            A series
        """
        try:
            megabyte = 1*10e+5
            for col in columns:
                self.df[col] = self.df[col] / megabyte
                self.df.rename(
                    columns={col: f'{col[:-7]}(MegaBytes)'}, inplace=True)

        except:
            print('failed to change values to megabytes')

        return self.df

    def fix_outlier(self, column: str):
        self.df[column] = np.where(self.df[column] > self.df[column].quantile(
            0.95), self.df[column].median(), self.df[column])

        return self.df[column]

    def standardized_column(self, columns: list, new_name: list, func) -> pd.DataFrame:
        try:
            assert(len(columns) == len(new_name))
            for index, col in enumerate(columns):
                self.df[col] = func(self.df[col])
                self.df.rename(columns={col: new_name[index]}, inplace=True)

        except AssertionError:
            print('size of columns and names provided is not equal')

        except:
            print('standardization failed')

        return self.df

    def optimize_df(self) -> pd.DataFrame:
        data_types = self.df.dtypes
        optimizable = ['float64', 'int64']
        for col in data_types.index:
            if(data_types[col] in optimizable):
                if(data_types[col] == 'float64'):
                    # downcasting a float column
                    self.df[col] = pd.to_numeric(
                        self.df[col], downcast='float')
                elif(data_types[col] == 'int64'):
                    # downcasting an integer column
                    self.df[col] = pd.to_numeric(
                        self.df[col], downcast='unsigned')

        return self.df.info()

    def save_clean_data(self, name: str):
        try:
            self.df.to_csv(name)

        except:
            print("Failed to save data")
