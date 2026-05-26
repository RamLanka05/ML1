import streamlit as st
import pandas as pd

st.title("My Data App")
uploaded_file = st.file_uploader("Choose a CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    st.line_chart(df)