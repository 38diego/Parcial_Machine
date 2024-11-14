import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.write(r'''
        La multicolinealidad es un fenómeno en la regresión lineal múltiple en el que dos o más variables predictoras 
        están altamente correlacionadas entre sí, lo que dificulta que el modelo distinga claramente el efecto individual 
        de cada variable sobre la variable dependiente. En términos más específicos, la multicolinealidad ocurre cuando 
        existe una relación lineal fuerte entre algunas de las variables predictoras, y esto crea redundancia en la información.

        Efectos de la multicolinealidad en la regresión lineal:

        - Inestabilidad en los coeficientes: Cuando las variables predictoras están altamente correlacionadas, los 
        coeficientes estimados de cada predictor se vuelven inestables. Pequeños cambios en los datos pueden 
        provocar grandes variaciones en los coeficientes.

        - Menor significancia estadística: Debido a la redundancia en la información aportada por las variables 
        correlacionadas, el modelo tiene dificultades para distinguir el efecto individual de cada variable 
        sobre la variable dependiente, lo que reduce su significancia estadística.

        - Aumento en la varianza de los coeficientes estimados: Esto disminuye la precisión de los intervalos de confianza.

        - Dificultades para interpretar el modelo: Dado que es complicado separar los efectos individuales de cada 
        predictor, puede ser difícil interpretar los resultados y entender cómo cada variable afecta realmente al resultado.

        Posibles soluciones en regresión lineal normal:

        - Eliminar variables: Si una variable no aporta información única, se puede considerar eliminarla del modelo.

        - Regularización: Métodos como Ridge o Lasso penalizan los coeficientes altos, lo que puede ayudar a reducir el impacto de la multicolinealidad.

        - Transformación de variables: Si las variables están relacionadas de manera predecible, la transformación puede reducir la multicolinealidad.

        - Análisis de componentes principales (PCA): El PCA puede convertir variables correlacionadas en un conjunto de nuevas variables 
        no correlacionadas, aunque pierde algo de interpretabilidad.

        Multicolinealidad en la regresión polinómica

        La multicolinealidad se presenta en los modelos de regresión polinómica debido a la correlación intrínseca entre las diferentes potencias 
        de una misma variable. Por ejemplo, en un modelo cuadrático:
        
        $y = \beta_0 + \beta_1 x + \beta_2 x^2$
         
        los términos suelen estar correlacionados $x$ y $x^2$, ya que ambos dependen de la misma variable original X. A medida que el modelo 
        incluye términos de mayor grado, como X3 o X4 , esta correlación aumenta, intensificando la multicolinealidad.
         
        2. Efectos de la multicolinealidad en la regresión polinómica

        Según el documento, los efectos de la multicolinealidad en la regresión polinómica son:

        - Inestabilidad en los coeficientes estimados: La presencia de multicolinealidad hace que el cálculo de los coeficientes a través del 
        método de mínimos cuadrados se vuelva inestable, lo que significa que pueden cambiar significativamente ante pequeñas variaciones en los datos.

        - Varianza elevada de los coeficientes: La matriz 𝑋𝑇𝑋, utilizada para calcular los coeficientes en la ecuación 𝑏=(𝑋𝑇𝑋)−1𝑋𝑇𝑌, se vuelve 
        casi singular (o no invertible) cuando hay multicolinealidad. Esto provoca un aumento en la varianza de los coeficientes, generando 
        estimaciones menos precisas.

        - Sobreajuste: La multicolinealidad puede llevar al modelo a ajustarse en exceso a las fluctuaciones de los datos de entrenamiento, 
        generando un modelo que no generaliza bien a nuevos datos.

        Posibles soluciones en regresión polinómica:

        - Regularización (Ridge o Lasso): La regularización es especialmente útil en la regresión polinómica para reducir el impacto de los términos 
        correlacionados.
        
        - Reducción de términos: En lugar de incluir todos los términos hasta un grado n, se puede probar con un modelo de menor grado.
        
        - Escalado de variables: Al estandarizar los valores antes de elevarlos a potencias, se puede reducir el impacto de la multicolinealidad.
         
        - Transformación de bases ortogonales: En vez de emplear términos polinomiales de x, se pueden usar bases ortogonales como los 
        polinomios de Chebyshev o de Legendre, que están diseñados para minimizar la multicolinealidad en modelos polinómicos.
         
        En el paper Minimizing the Effects of Collinearity in Polynomial Regression #link http://www.eng.tau.ac.il/~brauner/publications/IandEC_36_4405_97.pdf

        - Mayor Precisión y Validez Estadística: La transformación de los datos permitió la obtención de correlaciones más precisas y estadísticamente válidas. 
        Se demostró que al reducir la multicolinealidad, era posible utilizar un polinomio de mayor grado, lo que mejoraba la precisión del modelo.

        - Transformación: La transformación que lleva los valores de la variable independiente al rango [−1,1] fue la más efectiva en reducir la colinealidad. 
        Esta transformación resultó en el número de condición más bajo y permitió ajustar un polinomio de mayor grado sin problemas de multicolinealidad.
         
        - Uso de Polinomios Ortogonales: Los polinomios ortogonales fueron útiles para reducir los efectos de la colinealidad, ya que redujeron la interdependencia 
        entre los parámetros del modelo, proporcionando intervalos de confianza más claros. Esto ayudó a identificar cuáles términos en el polinomio eran 
        estadísticamente necesarios.

        - Límite de Orden del Polinomio: La relación entre el error de truncamiento y el error natural proporcionó un criterio para establecer el grado máximo de 
        polinomio que podía justificarse estadísticamente. Si el valor de esta relación era mayor que 1, se justificaba el uso de un polinomio de orden superior; 
        de lo contrario, la precisión adicional no agregaba valor estadístico al modelo.

         ''')


import statsmodels.api as sm

# Generación de datos
np.random.seed(23)
X = np.linspace(0, 10, 100)
y = 10 * np.exp(-0.2 * (X - 1)**2) - 5 * np.exp(-0.5 * (X - 6)**2) + np.random.normal(0, 1, X.shape)

col1, col2 = st.columns(2)

with col2: 
    scatter_plot = go.Figure()
    scatter_plot.add_trace(go.Scatter(x=X, y=y, mode='markers', marker=dict(color='blue')))
    scatter_plot.update_layout(
        title="Scatter Plot of Data",
        xaxis_title="X",
        yaxis_title="Y",
        height = 500
    )

    st.plotly_chart(scatter_plot)

with col1:
    kde_plot = ff.create_distplot([y], ["Density"], show_hist=False, show_rug=False)
    kde_plot.update_layout(title="Sapa",height = 500,showlegend = False)
    st.plotly_chart(kde_plot)

# Separar los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

def polynomial_regression(degree):
    X_new = np.linspace(0, 10, 100).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ("poly_features", poly_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    polynomial_regression.fit(x_train, y_train)
    y_new = polynomial_regression.predict(X_new)

    X_train_poly = poly_features.fit_transform(x_train)  # Transformar X_train a la forma polinómica
    X_train_with_intercept = sm.add_constant(X_train_poly)  # Agregar constante para OLS
    ols_model = sm.OLS(y_train, X_train_with_intercept).fit()
    
    # Gráfica de regresión polinómica con Plotly
    poly_plot = go.Figure()
    poly_plot.add_trace(go.Scatter(x=X_new.flatten(), y=y_new, mode='lines', name="Degree " + str(degree), line=dict(color='red')))
    poly_plot.add_trace(go.Scatter(x=x_train.flatten(), y=y_train, mode='markers', name="Train Data", marker=dict(color='blue')))
    poly_plot.add_trace(go.Scatter(x=x_test.flatten(), y=y_test, mode='markers', name="Test Data", marker=dict(color='green')))
    poly_plot.update_layout(
        title=f"Polynomial Regression (Degree {degree})",
        xaxis_title="X",
        yaxis_title="y"
    )
    st.plotly_chart(poly_plot)

    st.write(ols_model.summary())
# Selección del grado en Streamlit
degree = st.selectbox("Select the degree for Polynomial Regression", [i for i in range(1, 26)])
polynomial_regression(degree)


import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.special import chebyt


# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Crear y entrenar el modelo de regresión con polinomios de Chebyshev
def chebyshev_regression(degree):
    X_new = np.linspace(0, 10, 100).reshape(-1, 1)

    # Generar características de Chebyshev
    chebyshev_features_train = np.column_stack([chebyt(d)(x_train.flatten()) for d in range(degree + 1)])
    chebyshev_features_test = np.column_stack([chebyt(d)(x_test.flatten()) for d in range(degree + 1)])
    chebyshev_features_new = np.column_stack([chebyt(d)(X_new.flatten()) for d in range(degree + 1)])
    
    # Ajustar el modelo
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
    model.fit(chebyshev_features_train, y_train)
    y_new = model.predict(chebyshev_features_new)

    # Obtener los coeficientes del modelo
    coef = model.named_steps['lin_reg'].coef_

    # Agregar una constante para statsmodels (término independiente)
    X_train_with_intercept = sm.add_constant(chebyshev_features_train)
    
    # Ajuste de modelo usando statsmodels para obtener el resumen
    ols_model = sm.OLS(y_train, X_train_with_intercept).fit()

    # Gráfico de la regresión
    poly_plot = go.Figure()
    poly_plot.add_trace(go.Scatter(x=X_new.flatten(), y=y_new, mode='lines', name="Degree " + str(degree), line=dict(color='red')))
    poly_plot.add_trace(go.Scatter(x=x_train.flatten(), y=y_train, mode='markers', name="Train Data", marker=dict(color='blue')))
    poly_plot.add_trace(go.Scatter(x=x_test.flatten(), y=y_test, mode='markers', name="Test Data", marker=dict(color='green')))
    poly_plot.update_layout(
        title=f"Chebyshev Polynomial Regression (Degree {degree})",
        xaxis_title="X",
        yaxis_title="y"
    )
    st.plotly_chart(poly_plot)

    # Mostrar resumen en Streamlit
    st.write(ols_model.summary())


# Selección del grado en Streamlit
degree = st.selectbox("Select the degree for Chebyshev Polynomial Regression", [i for i in range(1, 26)])
chebyshev_regression(degree)

