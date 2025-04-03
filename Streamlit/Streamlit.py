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

st.write("Vous avez selectionné la demande n°", ID)

id=ID
id=int(id)

df_selection = df_[df_["SK_ID_CURR"]== id]
df_selection=df_selection.drop("SK_ID_CURR", axis=1)
#except (KeyError, TypeError):
    #print("Error: Application ID not found. Make sure the ID is correct.")

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.


st.sidebar.write("**Age du client :**", int(df_.iloc[id,2]/365), "ans")
st.sidebar.write("**Durée du crédit :**", int(df_.iloc[id,0]), "ans")
st.sidebar.write("**Montant de l'annuité :**", int(df_.iloc[id,7]), "$")
st.sidebar.write("**Charge du crédit par rapport au revenu :**", int(df_.iloc[id,11]*100), "%")
st.sidebar.write("**Ancienneté dans l'emploi :**", int(df_.iloc[id,4]/-365), "ans")













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


# Afficher les graphiques des variables:
 
st.sidebar.header("Plus d'informations")
st.sidebar.subheader("Visualisations univariées")
variables=['CREDIT_TERM','DAYS_BIRTH', "DAYS_EMPLOYED", "AMT_ANNUITY", "CREDIT_INCOME_PERCENT","ANNUITY_INCOME_PERCENT"]
features=st.sidebar.multiselect("les variables clés:", variables)
 
for feature in features:
         # Set the style of plots
         plt.style.use('fivethirtyeight')
         fig=plt.figure(figsize=(6, 6))
         #if feature=='DAYS_BIRTH':
         # Plot the distribution of feature
         st.write(feature)
         h1=plt.hist(df_[feature], edgecolor = 'k', bins = 25)
         plt.axvline(int(df_[feature][df_.index==id]), color="red", linestyle=":")
         plt.title(feature + " distribution", size=5)
         plt.xlabel(feature, size=5)
         plt.ylabel("Nombre d'observations", size=5)
         plt.xticks(size=5)
         plt.yticks(size=5)
         st.pyplot(fig)
