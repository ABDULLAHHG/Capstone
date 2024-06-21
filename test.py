import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import plotly.express as px
from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.regression import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment


with st.sidebar:
    st.header("The steps to get the algorithms prediction accuracy")
    st.text("1- upload csv file or excel file ")
    st.text("2- choose target feature")
    st.text("3- remove not important features ")
    

data = pd.DataFrame()
target = ""
def upload_dataset():
    dataset = st.file_uploader("upload csv file or excel file", type=['csv', 'xslx'])
    global data 
    if dataset is not None:
        if "csv" in dataset.name:
            data = pd.read_csv(dataset)
        if "xlsx" in dataset.name:
            data = pd.read_excel(dataset)
        st.write(data.head())
        st.write(data.shape)
        global target
        target = st.selectbox("choose The target variable?", data.columns)

def preprocessing_data():
    numerical_features = data.select_dtypes(['int64', 'float64']).columns
    categorical_feature = data.select_dtypes(['object']).columns
    missing_value_num = st.radio(
        "Set missing value for numerical value 👇",
        ["mean", "median"]
        
    )
    missing_value_cat = st.radio(
        "Set missing value for numerical value 👇",
        [ 'most frequent' ,"put additional class" ]
        
        
    )
    for col in numerical_features:
        data[col] = SimpleImputer(strategy=missing_value_num, missing_values=np.nan).fit_transform(data[col].values.reshape(-1, 1))
    for col in categorical_feature:
        if data[col].nunique() > 7 :
            data[col] = SimpleImputer(strategy='most_frequent', missing_values=np.nan).fit_transform(data[col].values.reshape(-1, 1))
        else:
            data[col] = LabelEncoder().fit_transform(data[col])
    if (len(numerical_features) != 0):
        st.header("Numerical Columns")
        st.write(numerical_features)
    if (len(categorical_feature) != 0):
        st.header("Categorical columns")
        st.write(categorical_feature)
    if (len(categorical_feature) != 0 or len(numerical_features) != 0):
        st.header("number of null value ")
        st.write(data.isna().sum())


def remove_feature():
    if len(target) != 0:
        select_columns = st.multiselect("select feature you want removed from dataframe",options=data.columns)
        data.drop(select_columns, axis=1, inplace=True)
        st.write(data.sample())
        
def compare_create_model():
    if(option == 'Regression'):
        s = RegressionExperiment()
    if(option == "Classification"):
        s = ClassificationExperiment()
    s.setup(data, target = target, session_id = 123)
    best = s.compare_models()
    st.text(best)
    st.header("best Algorithm")
    st.write(best)
    st.write(s.evaluate_model(best))
    st.header(" 30 row of Preduction ")
    predictions = s.predict_model(best, data=data, raw_score=True)
    st.write(predictions.head(30))
    
option = st.selectbox(
        " What would you like to use?",
        ("Regression", "Classification"),
        )


upload_dataset()
if (data.shape!=(0, 0)):
    remove_feature()
    preprocessing_data()
    compare_create_model()