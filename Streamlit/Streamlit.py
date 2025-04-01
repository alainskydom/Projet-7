import streamlit as st
import json
from typing import List
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

st.set_page_config(page_title='Credit Score Application', page_icon='💰', layout='wide',
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
df_ = pd.read_csv(r"Streamlit/df_api_1000.csv")
df_=df_.loc[:, ~df_.columns.str.match ('Unnamed')]
#df_calc= df_.drop(['TARGET', 'SK_ID_CURR'], axis=1)
# df.drop(columns='index', inplace=True)

# Define the threshold of for application.
threshold = 0.08

#---- SIDEBAR ----
st.sidebar.header("Merci de selectionner la demande de crédit:")
ID = st.sidebar.selectbox(
    "Choisissez l'identifiant du dossier:",
    options=df_["SK_ID_CURR"].unique()
)

#ID = st.sidebar.multiselect(
    #"Select the ID:",
    #options=df_["SK_ID_CURR"].unique(),
    #default=df_["SK_ID_CURR"].unique()
#)

st.write("Vous avez selectionné", ID)

id=ID

df_selection = df_[df_["SK_ID_CURR"]== id]
df_selection=df_selection.drop("SK_ID_CURR", axis=1)
#except (KeyError, TypeError):
    #print("Error: Application ID not found. Make sure the ID is correct.")

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.


st.sidebar.write("**Age du client :**", int(df_.iloc[id,2].values /-365), "ans")

run = st.button( 'Evaluer le score de crédit de la demande')

plt.text(0.7, 1.05, "POOR", horizontalalignment='left', size='medium', color='white', weight='semibold')
plt.text(2.5, 1.05, "REGULAR", horizontalalignment='left', size='medium', color='white', weight='semibold')
plt.text(4.7, 1.05, "GOOD", horizontalalignment='left', size='medium', color='white', weight='semibold')

placeholder = st.empty()

st.header('Résultats')

if run:
    #features = df_selection.to_numpy().reshape(1, -1)
    score = best_model.predict_proba(df_selection)[:, 1]
    #features = df_.loc[ID, df_calc.columns].to_numpy().reshape(1, -1)
    st.write("The Score of the Client is ", score)
  


    if score <= threshold :
        st.balloons()
        t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='red')
        placeholder.markdown('Votre demande de crédit est acceptée! Bravo!')
        st.markdown(
            'Compte tenu des informations produites par le client le risque de défaut est faible')
    elif score > threshold:
        t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='red')
        placeholder.markdown('Votre demande de crédit est refusée!')
        st.markdown(
            'Compte tenu des informations produites, il existe un risque significatif de défaut.')
    elif score == -1:
        t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='red')
        placeholder.markdown('Your credit score is **POOR**.')
        st.markdown(
            'This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
    plt.gca().add_patch(t1)
