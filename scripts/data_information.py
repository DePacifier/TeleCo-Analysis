import pandas as pd


class DataInfo:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_information(self):
        return self.df.info()

    def get_size(self):
        print(
            f"The DataFrame containes {self.db.shape[0]} rows and {self.db.shape[1]} columns.")
        return self.db.shape

    def get_total_entries(self):
        print(f"The DataFrame containes {self.db.size} entries.")
        return self.db.size

    def get_description(self):
        return self.df.describe()

    def get_column_description(self, column_name: str):
        try:
            return self.df[column_name].describe()
        except:
            print("Failed to get decription of the column")

    def get_mean(self):
        return self.df.mean()

    def get_column_mean(self, column_name: str):
        try:
            return self.df[column_name].mean()
        except:
            print("Failed to get decription of the column")

    def get_standard_deviation(self):
        return self.df.std()

    def get_column_standard_deviation(self, column_name: str):
        try:
            return self.df[column_name].std()
        except:
            print("Failed to get decription of the column")

    def get_total_missing_values(self):
        value = self.df.isnull().sum().sum()
        print(f"The total number of missing values is {value}")
        return value

    def get_columns_with_missing_values(self):
        lst = self.df.isnull().any()
        print(f"Columns with missing values are:\n{lst}")
        return lst

    def get_column_based_missing_values(self):
        value = self.df.isnull().sum()
        print(value)
        return value

    def get_memory_usage(self):
        print(
            f"The Dataframe is currently using {self.df.memory_usage()} bytes.")
        return self.df.memory_usage()

    def get_aggregate(self, stat_list: list):
        try:
            return self.df.agg(stat_list)
        except:
            print("Failed to get aggregates")

    def get_matrix_correlation(self):
        return self.df.corr()

    def get_grouped_by(self, column_name: str):
        try:
            return self.df.groupby(column_name)
        except:
            print("Failed to get grouping column")
