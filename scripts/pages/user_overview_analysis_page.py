import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, '../scripts')
from results_pickler import ResultPickler

def app():

    # Load Saved Results Data
    results = ResultPickler()
    results.load_data(file_name='./data/overview_results.pickle')

    st.title("User Overview analysis")

    st.header("Overall Handset Manufacturers and Type Data")
    st.subheader("Top 10 Handsets used by customers")
    st.dataframe(results.data['top_10_handsets'])

    st.subheader("Top 3 Handsets Manufacturers")
    st.dataframe(results.data['top_3_handset_manufacturers'])

    top_manu = results.data['top_3_handset_manufacturers'].index.tolist()

    st.subheader(f"Top 5 Handsets Made by {top_manu[0]}")
    st.dataframe(results.data['best_5_apple_handsets'])
    
    st.subheader(f"Top 5 Handsets Made by {top_manu[1]}")
    st.dataframe(results.data['best_5_samsung_handsets'])

    st.subheader(f"Top 5 Handsets Made by {top_manu[2]}")
    st.dataframe(results.data['best_5_huawei_handsets'])

    st.subheader("Top 5 Handsets Among The Top Producers")
    st.dataframe(results.data['best_5_handsets'])

    st.subheader("Top 5 Handsets Among The Top Producers")
    st.dataframe(results.data['best_5_handsets'])
