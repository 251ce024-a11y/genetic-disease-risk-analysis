
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Genetic Disease Risk Analysis", layout="wide")

st.title("Genetic Disease Risk Analysis")
st.write("PCA and LDA based Genetic Data Analysis")

@st.cache_resource
def load_models():
    return joblib.load("genetic_models.joblib")

try:
    models = load_models()

    st.success("Trained models loaded successfully!")

    st.subheader("Model Information")
    st.write("Available components:")
    st.write(list(models.keys()))

    st.subheader("PCA Explained Variance")
    try:
        df = pd.read_csv("pca_explained_variance.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.info("PCA variance CSV not found.")

    st.subheader("PCA Patient Coordinates")
    try:
        pca_df = pd.read_csv("pca_patient_coordinates.csv")
        st.dataframe(pca_df)
    except FileNotFoundError:
        st.info("PCA coordinates CSV not found.")

    st.subheader("LDA Patient Coordinates")
    try:
        lda_df = pd.read_csv("lda_patient_coordinates.csv")
        st.dataframe(lda_df)
    except FileNotFoundError:
        st.info("LDA coordinates CSV not found.")

except Exception as e:
    st.error(f"Error loading model: {e}")
