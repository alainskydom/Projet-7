import streamlit as st
import json
from typing import List
#from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import io
import os
import uuid
from zipfile import ZipFile
#import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Credit Score App', page_icon='💰', layout='wide',
                   initial_sidebar_state='auto', menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': '''
                        This app and its purpose is to asses the Credit Score of applicants based on their credit data. 

                        '''
    })


@st.cache(allow_output_mutation=True)
def load_model():
    try:
        return pickle.load(open(r"Streamlit/model.pkl", 'rb'))
    except FileNotFoundError:
        print("Error: Model file not found. Make sure 'model.pkl' is in the correct directory.")
        return None

best_model = load_model()


# Charger les données
df_ = pd.read_csv(r"Streamlit/df_api.csv")
df_=df_.loc[:, ~df_.columns.str.match ('Unnamed')]
df_calc= df_.drop(['TARGET', 'SK_ID_CURR'], axis=1)
# df.drop(columns='index', inplace=True)

# Define the threshold of for application.
threshold = 0.6

#---- SIDEBAR ----
st.sidebar.header("Please choose the application ID:")
ID = st.sidebar.radio(
    "Select the ID:",
    options=df_["SK_ID_CURR"].unique()
)

#ID = st.sidebar.multiselect(
    #"Select the ID:",
    #options=df_["SK_ID_CURR"].unique(),
    #default=df_["SK_ID_CURR"].unique()
#)

st.write("You selected:", ID)

id=ID

df_selection = df_[df_["SK_ID_CURR"]== id]
df_selection=df_selection.drop(['TARGET', 'SK_ID_CURR'], axis=1)
#except (KeyError, TypeError):
    #print("Error: Application ID not found. Make sure the ID is correct.")

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.

run = st.button( 'Assess credit application')

placeholder = st.empty()

st.header('Credit Score Results')

if run:
    features = df_selection.to_numpy().reshape(1, -1)
    score = best_model.predict_proba(features)[:, 1]
    #features = df_.loc[ID, df_calc.columns].to_numpy().reshape(1, -1)


    if score >= threshold :
        st.balloons()
        placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
        st.markdown(
            'This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
    elif score < threshold:
        placeholder.markdown('Your credit score is **REGULAR**.')
        st.markdown(
            'This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
    elif score == -1:
        t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
        placeholder.markdown('Your credit score is **POOR**.')
        st.markdown(
            'This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')

