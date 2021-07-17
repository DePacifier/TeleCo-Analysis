from numpy.core.records import array
import streamlit as st
from joblib import load
import numpy as np
import sys
sys.path.insert(0, '../scripts')


def app():

    # Load Saved Results Data
    model = load(filename='./models/satisfaction_scorer_model.joblib')

    st.title("Satisfaction Predictor Model")

    st.header("Insert Values to get the Satisfaction score of a user")
    st.subheader("How many Sessions User had: ")
    session_count = st.number_input('Enter sessions',key='a')
    
    st.subheader("How much total time did the User pass in hour: ")
    total_time = st.number_input('Enter total time', key='b')

    st.subheader("How much total Data did the User use in MegaBytes: ")
    total_data = st.number_input('Enter total data', key='c')

    st.subheader("How many TCP connections were retransmitted in Bytes: ")
    total_retransmission = st.number_input('Enter tcp retransmission',key='d')

    st.subheader("How much average delay the User had: ")
    average_delay = st.number_input('Enter average delay',key='e')

    st.subheader("How much average throughput the user had: ")
    total_throughput = st.number_input('Enter average throughput',key='f')

    if st.button('Predict Satisfaction'):
        try:
            array = [session_count, total_time, total_data,
                     total_retransmission, average_delay, total_throughput]
            val = model.predict([array])
            satisfaction = [i[0] for i in val][0]
            st.write(f'The predict satisfaction score of the user will be: {satisfaction}')
        except:
            print('Need more inputs')
