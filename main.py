import streamlit as st
import pandas as pd
from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import mean_squared_error
import numpy as np 

import plotly.graph_objects as go 
from plotly.subplots import make_subplots 

# pip install pycaret

# PyCaret's default installation will not install all the extra dependencies automatically. For that you will have to install the full version:

# pip install pycaret[full]


# loading sample dataset from pycaret dataset module
import streamlit as st
st.sidebar.title("Website Settings")

# These function taken from DataPrepKit they was a methods in it with some edit 
def read_data(file) -> pd.DataFrame :
    file_extension = file.name.rsplit(".", 1)[-1].lower()
    try:
        if file_extension == "csv":
            return pd.read_csv(file)
        elif file_extension == "excel":
            return pd.read_excel(file)
        elif file_extension == "json":
            return pd.read_json(file)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error reading data: {e}")

def handling_numerical_missing_values(column : pd.Series , startigay : str) -> pd.Series:
        if column.dtype == "int64":
            if startigay == "mod":
                return column.fillna(column.mode()[0])
            elif startigay == "mean":
                return column.fillna(column.mean())
            elif startigay == "median":
                return column.fillna(column.median())

        elif column.dtype == "float64":
            if startigay == "mod":
                return column.fillna(column.mode()[0])
            elif startigay == "mean":
                return column.fillna(column.mean())
            elif startigay == "median":
                return column.fillna(column.median())
            
def handle_catigorical_missing_values(column : pd.Series , stratigay : str) -> pd.Series:
    return column.fillna(column.mode()[0])

def encode_column(df , column_name: str, encoding_method: str) -> None:
    if encoding_method == "categorical":
        df[column_name] = df[column_name].astype("category")
        df[column_name] = df[column_name].cat.codes
    elif encoding_method == "one-hot": # 
        df = pd.get_dummies(df[column_name], columns=[column_name] ,drop_first=True)
    else:
        raise ValueError(f"Unsupported encoding method: {encoding_method}")
    return df

# vis function 

# Color for Pie plot and Bar plot 
colors = ['#7c90db', '#92a8d1', '#a5c4e1', '#f7cac9', '#fcbad3', '#e05b6f', '#f8b195', '#f5b971', '#f9c74f', '#ee6c4d', '#c94c4c', '#589a8e', '#a381b5', '#f8961e', '#4f5d75', '#6b5b95', '#9b59b6', '#b5e7a0', '#a2b9bc', '#b2ad7f', '#679436', '#878f99', '#c7b8ea', '#6f9fd8', '#d64161', '#f3722c', '#f9a828', '#ff7b25', '#7f7f7f']

# This project from my repo Simple EDA streamlit wtih some edit 
def subplot(df):
    column = st.selectbox("Choose a column to view its dist" ,df.columns ,int(np.argmin(df.nunique())))
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Countplot', 'percentage'), specs=[[{"type": "xy"}, {'type': 'domain'}]])

    # Bar plot
    fig.add_trace(
        go.Bar(
            x=df[column].value_counts().index,
            y=df[column].value_counts().values,
            textposition='auto',
            showlegend=False,
            marker=dict(
                color=colors[:len(df[column].value_counts())],  
                line=dict(color='black', width=2)
            )
        ),
        row=1,
        col=1
    )

    # Pie plot 
    fig.add_trace(
        go.Pie(
            labels=df[column].value_counts().index,
            values=df[column].value_counts().values,
            hoverinfo='label',
            textinfo='percent',
            textposition='auto',
            marker=dict(
                colors=colors[:len(df[column].value_counts())],  
                line=dict(color='black', width=2)
            )
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title = {'text' : f'Distribution of the {column}',
                 'y' : 0.9,
                 'x' : 0.5,
                 'xanchor' : 'center',
                  'yanchor' : 'top'},
                  template = 'plotly_dark')

    st.plotly_chart(fig)
    
def pie():
    catigorical = [col for col in df.select_dtypes(["object" , "category"]).columns if df[col].nunique() < 10]
    pass




# Sidebar options
upload_option = st.sidebar.radio("Choose Upload Option", ("One File", "Two Files"))

file_uploaded : bool = False

# Upload Dataset
if upload_option == "One File":
    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx" , "json"])
    if file is not None:
        st.success("File uploaded successfully!")
        file_uploaded = True

        # Read Data 
        df = read_data(file)
        subplot(df)

        # Make a user able to choose a Target column
        target = st.selectbox("choose The target variable", df.columns)

# Upload Datasets for Train and Test
elif upload_option == "Two Files":
    # still need a work 
    st.header("not supported for now  maybe one day will do it")
    file1 = st.file_uploader("Upload Train Dataset", type=["csv", "xlsx" , "json"])
    file2 = st.file_uploader("Upload Test Dataset", type=["csv", "xlsx" , "json"])
    if file1 is not None and file2 is not None:
        st.success("Files uploaded successfully!")
        file_uploaded = True
        
        # Read Data
        train = read_data(file1)
        test = read_data(file2)
        
        # Make a user able to choose a Target column
        target = st.selectbox("choose The target variable?", train.columns)


def Handle_missing_values():
    # Select the numerical columns and catigorical columns based on datatypes 
    numerical_columns = df.select_dtypes(['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(['object']).columns
    
    # Ask a user to choose a way to handle numerical missing values
    numerical_startigay = st.radio("Choose a way to handle numerical missing values",
                                    ["mean", "median","mod"])
    
    # Ask a user to choose a way to handle catigorical missing values
    catigorical_startigay = st.radio("Choose a way to handle catigorical missing values",
                                    ['mode'])
    
    # Handle numerical missing values 
    for col in numerical_columns:
        df[col] = handling_numerical_missing_values(df[col] , numerical_startigay )
    
    # Handle catigorical missing values 
    for col in categorical_columns:
        df[col] = handle_catigorical_missing_values(df[col] , catigorical_startigay)
    return df 

def Drop_columns(df):
    columns_to_drop = st.multiselect("choose columns to drop",df.columns)
    if len(columns_to_drop) > 0: 
        df = df.drop(columns_to_drop  , axis = 1)
    return df 

def DataEncoder(df : pd.DataFrame):
    # For label encoding 
    label_encoder_columns = st.multiselect("choose columns for Label Encoding method",df.select_dtypes(["object" , "category"]).columns, [col for col in df.select_dtypes(["object" , "category"]) if df[col].nunique() >= 5 ])
    
    # For catigorical encoding 
    catigorical_encoder_columns = st.multiselect("choose columns for Catigorical Encoding method",df.select_dtypes(["object" , "category"]).columns, [col for col in df.select_dtypes(["object" , "category"]) if df[col].nunique() < 5 ])
    
    # implementation for label encoder method 
    if len(label_encoder_columns)>0:
        for col in label_encoder_columns:
            df = df.join(encode_column(df , col , "one-hot" ))
            df = df.drop(col , axis = 1)

    # implementation for catigorical encoding method 
    if len(catigorical_encoder_columns)>0:
        for col in catigorical_encoder_columns:
            encode_column(df , col , "categorical")
    
    return df 

def showData(df):
    if df is not None:
        st.dataframe(df)

# 
def create_model_And_evaluate_model(df = None , train = None , test = None):
    if(SelectModelType == 'Regression'):
        model = RegressionExperiment()
    if(SelectModelType == "Classification"):
        model = ClassificationExperiment()
    else:
        st.text("Choose Model Type")
    
    if upload_option == "One File":
        y_true = df[target]

        model.setup(df, target = target, session_id = 123)
        best_model = model.compare_models()

        # Pull metrics
        metrics = model.pull()

        # Show best Models 
        st.header("best_model Models")
        st.write(metrics)

        # Choosing model 
        st.text((f"Choosing model : {metrics.Model[0]}"))
        
        # st.write(model.evaluate_model(metrics.index[0]))
        
        y_pred = model.predict_model(best_model, data=df)

        st.header(" 10 row of Prediction ")
        st.write(pd.concat([y_pred[[target]] , y_pred.iloc[:,-2:]]).sample(10))


    elif upload_option == "Two Files":
        y_true = test[target]

        model.setup(train, target = target, session_id = 123)
        best_model = model.compare_models()

        # Pull metrics
        metrics = model.pull()

        # Show best Models 
        st.header("best_model Models")
        st.write(metrics)
        st.text((f"Choosing model : {metrics.Model[0]}"))

        # st.write(model.evaluate_model(metrics.index[0]))

        y_pred = model.predict_model(best_model, data=test, raw_score=True)
        st.header(" 10 row of Prediction ")
        st.write(pd.join([y_pred[[target]] , y_pred[:,-2:]]).sample(10))

    if SelectModelType == "Regression":
        st.text(f"mean square error : {metrics.MSE[0]}")
        
        try:
            model.plot_model(best_model, plot = 'auc' , display_format="streamlit")
        except:
            st.warning("Area Under the Curve plot not work")

    elif SelectModelType == "Classification":
        model.plot_model(best_model , plot = 'confusion_matrix',display_format="streamlit", plot_kwargs = {'percent' : True})
        model.plot_model(best_model, plot = 'auc' , display_format="streamlit")
        
    else:
        st.text("Choose Model Type")





if file_uploaded:
    if upload_option == "One File":
        # Original dataset
        ShowOriginalDataset : bool = st.checkbox("Show Original dataset" , 1)
        if ShowOriginalDataset:
            showData(df)
        df = Drop_columns(df)
        df = Handle_missing_values()
        df = DataEncoder(df)
        # After Preprocessing
        ShowPreprocessingDataset : bool = st.checkbox("Show Preprocessing dataset" , 1)
        if ShowOriginalDataset:
            showData(df)
        SelectModelType : str = st.selectbox("Choose Model Type",("Regression", "Classification") , 0 if df[target].nunique() > int(df.shape[0] * 0.01) else 1)
        if st.button("Start"):
            create_model_And_evaluate_model(df = df , train = None , test = None)



    if upload_option == "Two Files": 
        selected : str = st.selectbox("Show Train or Test Data" , ["Train Data" , "Test Data"] , 0)
        SelectModelType : str = st.selectbox("Choose Model Type",("Regression", "Classification") , 0 if df[target].nunique() > int(train.shape[0] * 0.01) else 1)
        if st.button("Start"):
            create_model_And_evaluate_model(df = None , train = train , test = test)

        if selected == "Train Data":
            showData(train)
        elif selected == "Test Data":
            showData(test)
        else:
            st.text("None of datasets are selected")

else:
    st.text("Upload Your data")
    if upload_option != "Two Files":
        st.text("You can choose to select Train and Test data from sidebar")
    if upload_option == "Two Files":
        st.text("You can choose to select a Whole dataset from sidebar")
