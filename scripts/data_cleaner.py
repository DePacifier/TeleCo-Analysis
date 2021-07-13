import pandas as pd


class DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        self.df.drop(columns, axis=1, inplace=True)
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
