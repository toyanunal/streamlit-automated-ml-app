import streamlit as st
import pandas as pd
import pandas_profiling as pp
import os
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from PIL import Image


st.write("Hello World")

with st.sidebar:
    automl = Image.open('automl.png')
    st.image(automl)
    