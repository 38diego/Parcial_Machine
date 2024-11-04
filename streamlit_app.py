import streamlit as st
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from streamlit_navigation_bar import st_navbar
import os

st.set_page_config(layout="wide")

st.markdown('# :red[Métodos y parámetros en regresión lineal]')

st.write("<p style='font-size:25px;'>1. ¿Qué método utiliza la clase LinearRegression de \
         Scikit-learn por defecto para ajustar un modelo de regresión lineal?</p>", 
         unsafe_allow_html=True)

st.write("<p style='font-size:23px;'>En Scikit-learn para entrenar un modelo de regresion \
         lineal se hace mediante la clase LinearRegresion del modulo linear_model, esta clase \
         tiene los siguientes parametros:</p>", 
         unsafe_allow_html=True)

st.write("<p style='font-size:23px; color:#663EC7;'><strong>LinearRegression(</strong></p>",
         unsafe_allow_html=True)

_, col1 = st.columns([0.1,1])

with col1:

    st.write("<p style='font-size:20px; color:#1461DB;'><strong>fit_intercept:</strong> bool, default=True</p>\
            \
            \
            <p style='font-size:20px;'>Parametro para calcular el intercepto o no. \
            Si se establece en False, no se utilizará ningún intercepto en los cálculos pero \
            se espera que los datos estén centrados).</p>", unsafe_allow_html=True)

    st.write("<p style='font-size:20px; color:#1461DB;'><strong>copy_X:</strong> bool, default=True</p>\
            \
            \
            <p style='font-size:20px;'>Parametro para hacer una copia de los datos, Si se \
            marca como False se sobreescribiran los cambios en los datos originales</p>", 
            unsafe_allow_html=True)
    
    st.write("<p style='font-size:20px; color:#1461DB;'><strong>positive:</strong> bool, default=False</p>\
            \
            \
            <p style='font-size:20px;'>Cuando se establece en True, obliga a que los \
            coeficientes sean positivos. Esta opción sólo es compatible con matrices densas.</p>", 
            unsafe_allow_html=True)
    
    st.write("<p style='font-size:20px; color:#1461DB;'><strong>n_jobs:</strong> int, default=None</p>\
            \
            \
            <p style='font-size:20px;'>número de núcleos que el algoritmo debe utilizar para \
            procesar la tarea. Esto sólo proporcionará un aumento de velocidad en el caso de problemas \
            suficientemente grandes, esta opcion se positive se establece en True. Si el parametro es \
            None significa 1 a menos que esté en un contexto joblib.parallel_backend. -1 significa \
            utilizar todos los procesadores.</p>\
            \
            \
            <p style='font-size:20px;'>Si el parámetro positive es True, añade una restricción adicional \
            en el ajuste del modelo, y el cálculo se vuelve más complejo porque el algoritmo tiene que \
            asegurarse de que todas las soluciones cumplen con esta restricción, por lo que no puede usar \
            el método estándar de mínimos cuadrados ordinarios. En su lugar, necesita utilizar un \
            algoritmo de optimización restringida, como Least Squares with Non-Negative Constraints \
            (NNLS), que tiene un costo computacional mayor y por lo tanto se necesitan mas jobs.</p>",
            unsafe_allow_html=True)

st.write("<p style='font-size:23px; color:#663EC7;'><strong>)</strong></p>", unsafe_allow_html=True)



st.write("<p style='font-size:23px;'>Para ajustar el modelo se utiliza el método:</p>", unsafe_allow_html=True)

st.write("<p style='font-size:23px; color:#663EC7;'><strong>.fit(</strong></p>",
         unsafe_allow_html=True)

_, col1 = st.columns([0.1,1])

with col1:

    st.write("<p style='font-size:20px; color:#1461DB;'><strong>X:</strong> {array-like, sparse matrix} of shape (n_samples, n_features)</p>\
            \
            \
            <p style='font-size:20px;'>Un arreglo que contiene los datos del conjunto de entrenamiento.</p>", unsafe_allow_html=True)

    st.write("<p style='font-size:20px; color:#1461DB;'><strong>y:</strong> array-like of shape (n_samples,) or (n_samples, n_targets)</p>\
            \
            \
            <p style='font-size:20px;'>Un arreglo que contiene la(s) variable(s) objetivo.</p>", 
            unsafe_allow_html=True)
    
    st.write("<p style='font-size:20px; color:#1461DB;'><strong>sample_weight:</strong> array-like of shape (n_samples,), default=None</p>\
            \
            \
            <p style='font-size:20px;'>Un arreglo que especifica los pesos para cada distintas muestras de los datos.</p>", 
            unsafe_allow_html=True)

st.write("<p style='font-size:23px; color:#663EC7;'><strong>)</strong></p>", unsafe_allow_html=True)

model = LinearRegression()

code1 = '''
print("Hello world ...")
'''

st.code(code1, language='R')

process1 = subprocess.Popen(["/usr/bin/Rscript", "helloworld.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
result1 = process1.communicate()
st.write(result1)
