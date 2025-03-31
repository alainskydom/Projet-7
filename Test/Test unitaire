import pytest
import pickle
import pandas as pd
from lightgbm import LGBMClassifier

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

df=pd.read_csv('Test/df_api_100.csv')
       
def test_predict_refus():
  # Arrange
  loaded_model = pickle.load(open(r"Test/model.pkl", 'rb'))
  print(loaded_model)
          
  # Act
  df['pred']=loaded_model.predict_proba(df)[:, 1]
  Mean_refus= df['pred'][df['pred']>0.08].mean()
  # Assert
  assert Mean_refus <= 0.25
