import streamlit as st
import pandas as pd
import pandas_profiling as pp
import os
from streamlit_pandas_profiling import st_profile_report
from pycaret import classification
from pycaret import regression
from PIL import Image

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('./dataset.csv', index_col=None)

with st.sidebar:
    automl = Image.open('automl.png')
    st.image(automl)
    st.title("Automated ML App")
    choice = st.radio("Navigation", ['Upload', 'Profiling', 'Modelling', 'Download'])
    st.info("This app explores given dataset and creates best models.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    task = st.selectbox("Choose the Modelling Task", ['Classification', 'Regression'])
    target = st.selectbox("Choose the Target Column", df.columns)
    if st.button("Run Modelling"):
        if task == "Classification":
            clf = classification.setup(data=df, target=target) # silent=True
            setup_df = classification.pull()
            st.dataframe(setup_df)
            best_model = classification.compare_models()
            compare_df = classification.pull()
            st.dataframe(compare_df)
            classification.save_model(best_model, 'best_model')
        elif task == "Regression":
            reg = regression.setup(data=df, target=target)
            setup_df = regression.pull()
            st.dataframe(setup_df)
            best_model = regression.compare_models()
            compare_df = regression.pull()
            st.dataframe(compare_df)
            regression.save_model(best_model, 'best_model')

if choice == "Download":
    st.title("Download Trained Model")
    with open('best_model.pkl', 'rb') as f:
        st.download_button("Download Model", f, 'best_model.pkl')