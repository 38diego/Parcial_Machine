import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown('# :red[Laboratorio de supuestos]')

st.write('''<p style='font-size:23px;'>
        Se utilizara el dataset boston_houstin que se encuentra en la libreria keras.datasets para aplicar
        una regresion lineal y comprobar sus supuestos y aplicar algunas acciones correctivas para ver que
        efecto tienen en el modelo. El conjunto de boston_housing contiene las siguientes variables:
        </p>''', 
        unsafe_allow_html=True)

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

_,col1,_ = st.columns([0.3,1,0.3])

with col1:
    # Crear el DataFrame
    df = pd.DataFrame({
        "Variables": train_data.columns,
        "Descripcion": [
            "Tasa de criminalidad per cápita por ciudad.",
            "Proporción de terrenos residenciales destinados a lotes mayores de 25,000 pies cuadrados.",
            "Proporción de acres de negocios no minoristas por ciudad.",
            "Variable ficticia del Río Charles (1 si la zona limita con el río, 0 en caso contrario).",
            "Concentración de óxidos de nitrógeno (partes por cada 10 millones).",
            "Número promedio de habitaciones por vivienda.",
            "Proporción de unidades ocupadas por sus propietarios que fueron construidas antes de 1940.",
            "Media ponderada de las distancias a cinco centros de empleo en Boston.",
            "Índice de accesibilidad a autopistas radiales.",
            "Tasa de impuestos a la propiedad por cada $10,000 de valor total.",
            "Relación alumno-profesor por ciudad.",
            "1000(Bk - 0.63)^2, donde Bk es la proporción de personas negras en la ciudad.",
            "Porcentaje de la población de menor estatus socioeconómico.",
            "Valor mediano de las viviendas ocupadas por sus propietarios (en miles de dólares)."
        ]
    })

    # Función para aplicar color a una fila específica (en este caso la segunda)
    def highlight_row(x):
        color = 'background-color: #A3F0FF'
        df_color = pd.DataFrame('', index=x.index, columns=x.columns)
        df_color.iloc[-1] = color  # Colorea la segunda fila (índice 1)
        return df_color

    st.table(df.style.apply(highlight_row, axis=None))

st.write('''<p style='font-size:23px;'>
        Ajustar el modelo de regresión lineal con statsmodels
        </p>''', 
        unsafe_allow_html=True)

X = train_data.drop(columns=["medv"])
y = train_data["medv"]

X_train_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_train_sm).fit()

# Predict on the training set to calculate residuals
y_train_pred_sm = model_sm.predict(X_train_sm)
residuals_sm = model_sm.resid

st.code(f'''
import statsmodels.api as sm
import pandas as pd
        
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

X = train_data.drop(columns=["medv"])
y = train_data["medv"]

X_train_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_train_sm).fit()

model_sm.summary().tables[1]
''')

summary = model_sm.summary()

coef_table = summary.tables[1]  # El índice 1 tiene los coeficientes

coef_table_df = pd.DataFrame(coef_table)

# Convertir todas las celdas a cadenas de texto para asegurar compatibilidad
coef_table_df = coef_table_df.applymap(str)

# Configurar nombres de columnas de forma explícita
coef_table_df.columns = ["Variable", "Coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
coef_table_df = coef_table_df.drop(0)  # Eliminar fila de encabezado duplicado
coef_table_df.set_index("Variable", inplace=True)

# Asegurarse de que los valores en la columna "P>|t|" sean numéricos
coef_table_df["P>|t|"] = pd.to_numeric(coef_table_df["P>|t|"], errors='coerce')

# Aplicar color a los valores p en la columna "P>|t|"
styled_coef_table = coef_table_df.style.applymap(
    lambda val: 'background-color: #B0ED8B' if val < 0.05 else 'background-color: #ED8181' if pd.notnull(val) else '',
    subset=["P>|t|"]
)

_,col1,_ = st.columns([0.2,1,0.2])

with col1:
    st.table(styled_coef_table)

st.write('''<p style='font-size:23px;'>
        A primera vista parece que el modelo podria ser bueno, pues solo 2 coeficientes no son significativos, ahora 
        miremos todos los supuestos para evaluar la validez del modelo: 
        </p>''', 
        unsafe_allow_html=True)

st.write('''<p style='font-size:23px;'><b>
        (a) Supuesto de media cero:</b> Graficar los residuos ei contra los valores predichos yi y comprobar si los residuos
        están distribuidos aleatoriamente alrededor de la línea de cero. 
        </p>''', 
        unsafe_allow_html=True)

st.code('''
residuals_sm = model_sm.resid

plt.figure(figsize=(10, 6))
sns.histplot(residuals_sm)
plt.axvline(np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
plt.show()
''')

plt.figure(figsize=(10, 6))
sns.histplot(residuals_sm)
plt.axvline(np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
st.pyplot(plt)

st.write('''<p style='font-size:23px;'><b>
        (b) Supuesto de homocedasticidad:</b> Verificar si los residuos muestran una varianza constante a lo largo de
        los valores predichos, es decir, si no forman patrones visibles como conos o parábolas. 
        </p>''', 
        unsafe_allow_html=True)

st.code('''
y_train_pred_sm = model_sm.predict(X_train_sm)
        
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred_sm, y=residuals_sm)
plt.axhline(y=np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
plt.show()
''')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred_sm, y=residuals_sm)
plt.axhline(y=np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
st.pyplot(plt)

st.write('''<p style='font-size:23px;'><b>
        (c) Supuesto de independencia:</b> Graficar los residuos en función del orden de las 
        observaciones para comprobar si hay dependencia entre ellos. 
        </p>''', 
        unsafe_allow_html=True)

st.code('''
plt.figure(figsize=(10, 6))
plt.plot(residuals_sm, marker='o', linestyle='-', color='b')
plt.axhline(y=np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
plt.show()
''')

plt.figure(figsize=(10, 6))
plt.plot(residuals_sm, marker='o', linestyle='-', color='b')
plt.axhline(y=np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
st.pyplot(plt)

st.write('''<p style='font-size:23px;'><b>
        (D) Supuesto de normalidad:</b> Realizar un gráfico Q-Q (quantile-quantile) para comparar 
        la distribución de los residuos con una distribución normal teórica. 
        </p>''', 
        unsafe_allow_html=True)

st.code('''
plt.figure(figsize=(10, 6))
sm.qqplot(residuals_sm, line='s',)
plt.title("Q-Q plot of residuals")
plt.show()
''')

plt.figure(figsize=(10, 6))
sm.qqplot(residuals_sm, line='s',)
plt.title("Q-Q plot of residuals")
st.pyplot(plt)

st.write('''<p style='font-size:23px;'><b>
        (e) Supuesto de ausencia de multicolinealidad:</b> Calcular el factor de inflación de la varianza (VIF) para
        cada variable predictora y verificar si algún valor es superior a 10, lo cual indicaría multicolinealidad. 
        </p>''', 
        unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.table(vif_data)

with col2:
    filtered_corr = train_data.corr().T
            
    # Crear una máscara para la parte superior diagonal y la diagonal principal
    mask = np.triu(np.ones_like(filtered_corr, dtype=bool), k=0)

    # Aplicar la máscara para ocultar la parte superior y la diagonal principal
    masked_corr = np.where(mask, np.nan, filtered_corr)

    # Crear una matriz de texto para el hover, con detalles
    text_array = np.array([f'{filtered_corr.columns[i]} vs {filtered_corr.columns[j]}: {filtered_corr.iloc[i, j]:.2f}'
    for i in range(len(filtered_corr.columns)) for j in range(len(filtered_corr.columns))])
    masked_text = np.where(mask, '', text_array.reshape(filtered_corr.shape))

    # Crear la figura con Plotly
    fig5 = go.Figure(data=go.Heatmap(
        z=masked_corr,
        x=filtered_corr.columns,
        y=filtered_corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        showscale=True,
        text=masked_text,
        texttemplate='',  # No mostrar texto en las celdas
        hoverinfo='text'  # Solo mostrar la información de hover
    ))

    fig5.update_layout(
        title="Matriz de Correlación (Parte Inferior)",
        height=500,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(filtered_corr.columns))),
            ticktext=filtered_corr.columns,
            title='Variables'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(filtered_corr.columns))),
            ticktext=filtered_corr.columns,
            title='Variables',
            autorange='reversed'  # Para que el eje Y tenga el mismo orden que el eje X
        )
    )

    st.plotly_chart(fig5)



'''
1. ¿Cómo se refleja el cumplimiento del supuesto de media cero en la distribución de los residuos y qué implicaciones
tiene para la precisión del modelo? Justifica tu respuesta detalladamente con base en los resultados obtenidos.


2. ¿Qué evidencia gráfica puedes observar para confirmar si se cumple el supuesto de homocedasticidad en los residuos
y qué consecuencias tendría la falta de varianza constante? Justifica con base en los patrones observados en las
visualizaciones.


3. ¿Existe algún patrón de dependencia en los residuos cuando se grafican según el orden de las observaciones y
qué impacto tiene en la validez del modelo? Justifica si el modelo puede estar fallando en capturar relaciones
importantes en los datos.


4. ¿Siguen los residuos una distribución normal según el gráfico Q-Q y cómo afecta esto a la validez de las pruebas de
hipótesis y los intervalos de confianza del modelo? Justifica tu respuesta con base en la alineación de los residuos
en el gráfico.


5. ¿Qué indican los valores de VIF sobre la multicolinealidad entre las variables predictoras y cómo afecta esto
la estabilidad y fiabilidad de los coeficientes? Justifica si es necesario aplicar alguna técnica para reducir la
multicolinealidad.


6. Si se detecta falta de independencia en los residuos, ¿qué ajustes o modificaciones podrían realizarse en el modelo
para corregir este problema? Justifica qué modificaciones serían adecuadas y por qué.


7. En caso de encontrar heterocedasticidad en los residuos, ¿qué estrategias podrías implementar para mejorar el
ajuste del modelo y garantizar una varianza constante? Justifica cada estrategia que propongas con base en el
análisis de los resultados.
'''