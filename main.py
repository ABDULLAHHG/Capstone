import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st


# PyCaret's default installation will not install all the extra dependencies automatically. For that you will have to install the full version:

# pip install pycaret[full]


# loading sample dataset from pycaret dataset module
from pycaret.datasets import get_data
from pycaret.classification import *


data = get_data('diabetes')
import streamlit as st
st.sidebar.title("Website Settings")

# Sidebar options
upload_option = st.sidebar.radio("Choose Upload Option", ("One File", "Two Files"))

if upload_option == "One File":
    file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    if file is not None:
        st.success("File uploaded successfully!")
        # Process the uploaded file

elif upload_option == "Two Files":
    file1 = st.file_uploader("Upload File 1", type=["csv", "xlsx"])
    file2 = st.file_uploader("Upload File 2", type=["csv", "xlsx"])
    if file1 is not None and file2 is not None:
        st.success("Files uploaded successfully!")
        # Process the uploaded files


st.dataframe(data)


clf1 = setup(data = data, 
             target = 'Class variable',
             numeric_imputation = 'mean',
            #  categorical_features = ['Sex','Embarked'], 
            #  ignore_features = ['Name','Ticket','Cabin'],
            #  silent = True
             )
st.dataframe(compare_models())


# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
lr = create_model('lr')

# plot model
st.pyplot(plot_model(lr, plot = 'auc'))
