import pandas as pd


def load_df_from_csv(filename: str) -> pd.DataFrame:
    """A simple function which tries to load a dataframe from a specified .csv filename"""
    try:
        df = pd.read_csv(filename)
        return df
    except:
        print("Error Occured:\n\tCould not find specified .csv file")


def load_df_from_excel(filename: str) -> pd.DataFrame:
    """A simple function which tries to load a dataframe from a specified .xslx filename"""
    try:
        df = pd.read_excel(filename, na_values=['?', None], engine='openpyxl')
        return df
    except:
        print("Error Occured:\n\tCould not find specified .xslx file")
