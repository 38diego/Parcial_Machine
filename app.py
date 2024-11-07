import streamlit as st

st.set_page_config(layout="wide")  # Debe ser la primera llamada

# --- Configuración de la navegación ---
Regresion_lineal = st.Page(
    'pages/Regresion_Lineal.py',
    title='Regresion_Lineal',
    url_path='',
    default=True,
)

Laboratorio = st.Page(
    'pages/Laboratorio_supuestos.py',
    title='Laboratorio_supuestos',
    url_path='Laboratorio',
)

streamlit_page = st.navigation([Regresion_lineal,Laboratorio])

streamlit_page.run()