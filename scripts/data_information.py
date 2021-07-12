import pandas as pd


class DataInfo:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_basic_description(self):
        self.get_size()
        self.get_total_memory_usage()
        self.get_memory_usage()
        self.get_information()

    def get_missing_description(self):
        self.get_total_missing_values()
        self.get_columns_with_missing_values()
        self.get_column_based_missing_values()

    def get_columns(self):
        print("Columns Listed in the DataFrame are: ")
        return self.df.columns.tolist()

    def get_information(self):
        print("DataFrame Information: ")
        return self.df.info()

    def get_size(self):
        value = self.df.shape
        print(
            f"The DataFrame containes {value[0]} rows and {value[1]} columns.")
        return value

    def get_total_entries(self):
        value = self.df.size
        print(f"The DataFrame containes {value} entries.")
        return value

    def get_description(self):
        print("DataFrame Description: ")
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
        missing_entries = self.df.isnull().sum().sum()
        total_entries = self.df.size
        print(f"The total number of missing values is {missing_entries}")
        print(round(((missing_entries/total_entries) * 100), 2),
              "%", "missing values.")
        return missing_entries

    def get_columns_with_missing_values(self):
        lst = self.df.isnull().any()
        # print(f"Columns with missing values are:\n{lst}")
        return lst.index

    def get_column_based_missing_values(self):
        value = self.df.isnull().sum()
        return value

    def get_column_based_missing_percentage(self):
        col_null = self.get_column_based_missing_values()
        total_entries = self.df.shape[0]
        missing_percentage = []
        for col_missing_entries in col_null:
            value = str(
                round(((col_missing_entries/total_entries) * 100), 2)) + " %"
            missing_percentage.append(value)

        missing_df = pd.DataFrame(col_null, columns=['total_missing_values'])
        missing_df['missing_percentage'] = missing_percentage
        return missing_df

    def get_columns_missing_percentage_greater_than(self, num: float):
        all_cols = self.get_column_based_missing_percentage()
        extracted = all_cols['missing_percentage'].str.extract(r'(.+)%')
        return extracted[extracted[0].apply(lambda x: float(x) >= num)].index

    def get_total_memory_usage(self):
        value = self.df.memory_usage(deep=True).sum()
        print(f"Current DataFrame Memory Usage:\n{value}")
        return value

    def get_memory_usage(self):
        print(f"Current DataFrame Memory Usage of columns is :")
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
