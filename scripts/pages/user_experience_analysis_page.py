import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, '../scripts')

from results_pickler import ResultPickler

def app():

    # Load Saved Results Data
    results = ResultPickler()
    results.load_data(file_name='./data/experience_results.pickle')

    st.title("User Experience analysis")

    st.header("Most Occuring Values of Metrics")
    st.dataframe(results.data['most_occuring_values'])

    st.header("Minimum and Maximum Values of Metrics")
    st.subheader("TCP retransmission (Bytes)")
    st.dataframe(results.data['min_max_10_TCP'])

    st.subheader("Avg Delay (ms)")
    st.dataframe(results.data['min_max_10_RTT'])

    st.subheader("Avg Throughput (kbps)")
    st.dataframe(results.data['min_max_10_Throughput'])

    st.header("Distribution of Metrics Based on Handset Type")
    st.subheader("By Avg Throughput (kbps)")
    st.subheader("Top:")
    st.dataframe(results.data['top_10_handsets_Throughput'])

    st.subheader("Least:")
    st.dataframe(results.data['least_10_handsets_Throughput'])

    st.subheader("By TCP retransmission (Bytes)")
    st.subheader("Top:")
    st.dataframe(results.data['top_10_handsets_TCP'])

    st.subheader("Least:")
    st.dataframe(results.data['least_10_handsets_TCP'])

    st.header("User Clusters")
    st.image('./data/experience_cluster.png')
