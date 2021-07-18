import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler


class DataManipulator:
    def __init__(self, df: pd.DataFrame):
        """
            Returns a DataManipulator Object with the passed DataFrame Data set as its own DataFrame
            Parameters
            ----------
            df:
                Type: pd.DataFrame

            Returns
            -------
            None
        """
        self.df = df

    def sort_using_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrame sorted with the specified column, default dataframe sorting
            Parameters
            ----------
            column:
                Type: str

            Returns
            -------
            pd.DataFrame
        """
        try:
            return self.df.sort_values(column)
        except:
            print("Failed to sort using the specified column")

    def get_top_sorted_by_column(self, column: str, length: int) -> pd.DataFrame:
        """
            Returns the objects DataFrame sorted in descending order and selecting the top ones with the specified column
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            pre_df = self.df.sort_values(
                column, ascending=False).iloc[:length, :]
            return pd.DataFrame(pre_df.loc[:, column])
        except:
            print("Failed to sort using the specified column and get the top results")

    def scale_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column scaled using MinMaxScaler
            Parameters
            ----------
            column:
                Type: str

            Returns
            -------
            pd.DataFrame
        """
        try:
            scale_column_df = pd.DataFrame(self.df[column])
            scale_column_values = scale_column_df.values
            print(
                f'The max and min values of the scaled {column} column are:\n\tmax: {scale_column_df.iloc[:, 0].min()}\n\tmin: {scale_column_df.iloc[:, 0].max()}')
            min_max_scaler = MinMaxScaler()
            scaled_values = min_max_scaler.fit_transform(scale_column_values)
            self.df[column] = scaled_values

            return self.df

        except:
            print("Failed to scale the column")

    def normalize_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            scale_column_df = pd.DataFrame(self.df[column])
            scale_column_values = scale_column_df.values
            normalizer = Normalizer()
            normalized_data = normalizer.fit_transform(scale_column_values)
            self.df[column] = normalized_data

            return self.df

        except:
            print("Failed to normalize the column")

    def standardize_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            std_column_df = pd.DataFrame(self.df[column])
            std_column_values = std_column_df.values
            standardizer = StandardScaler()
            normalized_data = standardizer.fit_transform(std_column_values)
            self.df[column] = normalized_data

            return self.df
        except:
            print("Failed to standardize the column")
