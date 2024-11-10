import streamlit as st
import subprocess
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from streamlit_navigation_bar import st_navbar
from PIL import Image
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, SGDRegressor

st.markdown('# :red[Métodos y parámetros en regresión lineal]')

### 1 punto
st.write('''
        <p style='font-size:25px;'><b>
        1. ¿Qué método utiliza la clase LinearRegression de Scikit-learn por 
        defecto para ajustar un modelo de regresión lineal?
        </b></p>''', 
        unsafe_allow_html=True)

st.write('''<p style='font-size:23px;'>
        El metodo usado por defecto es el de Minimos cuadrados ordinarios (OLS) cuando usamos
        la clase LinearRegression, un ejemplo de su uso:
        </p>''', 
        unsafe_allow_html=True)
        

st.code('''
from sklearn.linear_model import LinearRegression # Existe un modulo unicamente para modelos lineales :0

#Generamos datos aleatorios
X = np.random.normal(loc=5, scale=2, size=100)

#Modelamos la variable objetivo con ruido
y = 3 + 1.5 * x + np.random.normal(scale=1, size=n)

#Asignar pesos altos: 100 si x > 5, 0.01 en caso contrario
weights = np.where(x > 5, 100, 0.01)

#Instanciamos un objeto de la clase LinearRegression
model = LinearRegression() 

#Metodo para entrenar, tecnica utilizada para entrenar = (OLS)
model.fit( 
        
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
        #Un arreglo que contiene los datos del conjunto de entrenamiento.

        y: array-like of shape (n_samples,) or (n_samples, n_targets)
        #Un arreglo que contiene la(s) variable(s) objetivo.

        sample_weight: array-like of shape (n_samples,), default=None
        #Un arreglo que especifica los pesos para cada distintas muestras de los datos.
)

#Ajustar el modelo sin ponderación 
model_sin_ponderacion = LinearRegression()
model_sin_ponderacion.fit(X.reshape(-1, 1), y)

#Ajustar el modelo con ponderación
model_con_ponderacion = LinearRegression()
model_con_ponderacion.fit(X.reshape(-1, 1), y, sample_weight=weights)

data = pd.DataFrame({'weights': weights})

data['highlight'] = np.where(data['weights'] == 100, 'Peso Alto', 'Peso Bajo')

highlight_colors = data['highlight'].map({'Peso Alto': 'red', 'Peso Bajo': 'gray'})

#Graficamos
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, 
    y=y,
    mode='markers',
    marker=dict(color=highlight_colors, opacity=0.7),
    name='Datos'
))

x_vals = np.linspace(-1, 10, 100)
y_vals_sin_ponderacion = model_sin_ponderacion.predict(x_vals.reshape(-1, 1))

fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals_sin_ponderacion,
    mode='lines',
    line=dict(color='blue', dash='dash', width=2),
    name='Sin Ponderación'
))

y_vals_con_ponderacion = model_sin_ponderacion.predict(x_vals.reshape(-1, 1))
fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals_con_ponderacion,
    mode='lines',
    line=dict(color='red', width=2),
    name='Con Ponderación'
))

fig.update_layout(
    title='Comparación de Regresión: Sin Ponderación vs. Con Ponderación',
    title_font=dict(size=18, family='Arial, sans-serif'),
    xaxis_title='Variable Independiente (x)',
    yaxis_title='Variable Dependiente (y)',
    legend=dict(
        orientation='h',  
        yanchor='bottom',
        y=-0.2, 
        xanchor='center',
        x=0.5
    ),
    font=dict(family="Arial, sans-serif", size=12),
    showlegend=True,
    template="plotly_white",
    height = 600,
    xaxis=dict(
            showgrid=True,  
            gridcolor='lightgray',  
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,  
            gridcolor='lightgray',  
            gridwidth=1
        )
)
        
fig.show()
''', language= 'python')

np.random.seed(123)

n = 100
mean = 5
sd_x = 2

# Generar la variable 'x' con una distribución normal
x = np.random.normal(loc=mean, scale=sd_x, size=n)

# Generar la variable 'y' en base a la fórmula: y = 3 + 1.5 * x + ruido normal
y = 3 + 1.5 * x + np.random.normal(scale=1, size=n)

# Asignar pesos más extremos: 100 si x > 5, 0.01 en caso contrario
weights = np.where(x > 5, 100, 0.01)

# Ajustar el modelo sin ponderación
model_sin_ponderacion = LinearRegression()
model_sin_ponderacion.fit(x.reshape(-1, 1), y)

# Ajustar el modelo con ponderación
model_con_ponderacion = LinearRegression()
model_con_ponderacion.fit(x.reshape(-1, 1), y, sample_weight=weights)

# Crear un DataFrame con los datos y las predicciones
data = pd.DataFrame({'weights': weights})

# Crear una nueva variable para resaltar los puntos con pesos altos
data['highlight'] = np.where(data['weights'] == 100, 'Peso Alto', 'Peso Bajo')

_, col1, _ = st.columns([0.1,0.2,0.1])

highlight_colors = data['highlight'].map({'Peso Alto': 'red', 'Peso Bajo': 'gray'})

# Crear la figura con Plotly
fig = go.Figure()

# Agregar la dispersión de los puntos
fig.add_trace(go.Scatter(
    x=x, 
    y=y,
    mode='markers',
    marker=dict(color=highlight_colors, opacity=0.7),
    name='Datos'
))

# Agregar la línea de regresión sin ponderación
x_vals = np.linspace(-1, 10, 100)
y_vals_sin_ponderacion = model_sin_ponderacion.predict(x_vals.reshape(-1, 1))
fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals_sin_ponderacion,
    mode='lines',
    line=dict(color='blue', dash='dash', width=2),
    name='Sin Ponderación'
))

# Agregar la línea de regresión con ponderación
fig.add_trace(go.Scatter(
    x=x_vals,
    y=model_con_ponderacion.predict(x_vals.reshape(-1, 1)),
    mode='lines',
    line=dict(color='red', width=2),
    name='Con Ponderación'
))

fig.update_layout(
    title='Comparación de Regresión: Sin Ponderación vs. Con Ponderación',
    title_font=dict(size=18, family='Arial, sans-serif'),
    xaxis_title='Variable Independiente (x)',
    yaxis_title='Variable Dependiente (y)',
    legend=dict(
        orientation='h',  # Orientación horizontal
        yanchor='bottom',
        y=-0.2,  # Colocar debajo del gráfico
        xanchor='center',
        x=0.5
    ),
    font=dict(family="Arial, sans-serif", size=12),
    showlegend=True,
    template="plotly_white",
    height = 600,
    xaxis=dict(
            showgrid=True,  # Mostrar la grilla en el eje X
            gridcolor='lightgray',  # Color de la grilla
            gridwidth=1  # Grosor de la línea de la grilla
        ),
        yaxis=dict(
            showgrid=True,  # Mostrar la grilla en el eje Y
            gridcolor='lightgray',  # Color de la grilla
            gridwidth=1  # Grosor de la línea de la grilla
        )
)

with col1:
        st.plotly_chart(fig)        
        st.caption('''**Figure 1.** Se asignaron pesos mayores a los datos se asignarion si tienen un valor a 
                   5 que son los que corresponden al color rojo''')


st.code(f'''
#Algunos atributos de esta clase son:

model_con_ponderacion.coef_ -> shape (n_features, ) or (n_targets, n_features)
#Este atributo almacena los coeficientes estimados de la regresión lineal, es decir, los 
#valores de las pendientes para cada característica (variable independiente) en el modelo ajustado.

>>> {model_con_ponderacion.coef_} ### Solo los coeficientes, no el intercepto

model_con_ponderacion.intercept_ -> float or array of shape (n_targets,)
#Este es el término independiente o intercepto del modelo, Si fit_intercept = False, este valor se establece en 0.0

>>> {model_con_ponderacion.coef_}

model_con_ponderacion.singular_ -> array of shape (min(X, y),)
#Los valores singulares se pueden usar para detectar problemas de colinealidad entre las características. Si algunos de estos valores son 
#cercanos a cero, puede indicar que una o más características son casi linealmente dependientes.

>>> {model_con_ponderacion.singular_}


model_con_ponderacion.n_features_in_ -> int
#Indica el número de características o variables independientes que el modelo ha visto durante el ajuste 
 
>>> {model_con_ponderacion.n_features_in_}
        
model_con_ponderacion.rank_ -> int
#Este atributo representa el rango de la matriz de características X, que es una medida de la independencia lineal de las columnas de X

>>> {model_con_ponderacion.rank_} #Si este numero es menor a el atributo .n_features_in_, indica multicolinealidad        
''',language = "python")

### 2 punto
st.write('''<p style='font-size:25px;'><b>
        2. En R, cuando utilizamos la función lm() para ajustar un modelo de regresión lineal,
        ¿qué técnica se emplea paracalcular los coeficientes?
        </b></p>''', unsafe_allow_html=True)

st.markdown('''
        <p style='font-size:23px;'>
        En R al usar lm() usa la tecnica de minimos cuadrados ordinarios (OLS) cuando no se indican pesos en el parametro weights, 
        Si este parametro tiene pesos, usa la tecnica de minimos cuadrados ponderados (WLS) con los pesos indicados en este parametro, 
        El modelo no necesita de indicar un entrenamiento, al hacer model <- lm(...) y ajustar lo que queramos, el modelo se entrena 
        de inmediato y usa la tecnica de Minimos cuadrados si no se indicaron pesos, si se indicaron pesos se usa minimos cuadrados 
        ponderados con los respectivos pesos, un ejemplo de uso y aplicacion de los pesos son los siguientes:
        </p>''', unsafe_allow_html=True)

st.code('''
library(ggplot2)

set.seed(123)
n <- 100
x <- rnorm(n, mean = 5, sd = 2)
y <- 3 + 1.5 * x + rnorm(n, sd = 1)

weights <- ifelse(x > 5, 100, 0.01)
        
# Ajustar el modelo sin ponderación, tecnica = OLS
modelo_sin_ponderacion <- lm(y ~ x)

# Ajustar el modelo con ponderación, tecnica = WLS
modelo_con_ponderacion <- lm(y ~ x, weights = weights)

datos <- data.frame(x = x, y = y, weights = weights)
datos$pred_sin_ponderacion <- predict(modelo_sin_ponderacion)
datos$pred_con_ponderacion <- predict(modelo_con_ponderacion)

datos$highlight <- ifelse(datos$weights == 100, "Peso Alto", "Peso Bajo")

ggplot(datos, aes(x = x, y = y)) +
  geom_point(aes(color = highlight), alpha = 0.7) +  
  geom_line(aes(y = pred_sin_ponderacion, color = "Sin Ponderación"), size = 1, linetype = "dashed") +
  geom_line(aes(y = pred_con_ponderacion, color = "Con Ponderación"), size = 1) +
  labs(
    title = "Comparación de Regresión: Sin Ponderación vs. Con Ponderación",
    x = "Variable Independiente (x)",
    y = "Variable Dependiente (y)",
    color = "Modelo"
  ) +
  scale_color_manual(
    name = "",
    values = c("Sin Ponderación" = "blue", "Con Ponderación" = "red", "Peso Alto" = "red", "Peso Bajo" = "gray")
  ) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(size = guide_legend(title = "Pesos"))     
''')

#process1 = subprocess.Popen(["Rscript", "lm.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#result1 = process1.communicate()
image = Image.open('plot.png')

_, col1, _ = st.columns([0.1,0.2,0.1])

with col1:
        st.image(image,width=600)
        st.caption('''**Figure 2.** Se asignaron pesos mayores a los datos se asignarion si tienen un valor a 
                   5 que son los que corresponden al color rojo''')

st.write('''<p style='font-size:25px;'><b>
        3. ¿Qué método está diseñado en Scikit-learn para aplicar el descenso del gradiente en la regresión lineal y 
        cuándo es útil utilizarlo?
        </b></p>''', 
        unsafe_allow_html=True)

st.write('''<p style='font-size:23px;'>
        El método SGDRegressor permite usar una implementación del algoritmo de descenso de gradiente estocástico (SGD)
        esta es una versión del descenso de gradiente clásico en el que en lugar de usar toda la base de datos para calcular 
        el gradiente en cada paso, se utiliza solo un subconjunto de los datos. Esto permite que el proceso sea mucho más rápido
        y adecuado para grandes conjuntos de datos, un ejemplo de uso y su comparacion frente al metodo clasico (OLS) es: 
        </p>''', unsafe_allow_html=True)

st.code('''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Generar datos de ejemplo (con ruido)
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Modelo de Mínimos Cuadrados Ordinarios (OLS)
ols = LinearRegression()
ols.fit(X, y)

# Predicciones de OLS
y_pred_ols = ols.predict(X)

sgd = SGDRegressor(max_iter=1, tol=1e-3, random_state=42, warm_start=True)

frames = []

for i in range(30):
    sgd.partial_fit(X, y)  
    y_pred_sgd = sgd.predict(X)
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=np.linspace(-3, 2, 100), 
                       y=sgd.predict(np.linspace(-3, 2, 100).reshape(-1, 1)),
                       mode='lines', 
                       name=f'SGD - Iteración {i+1}', 
                       line=dict(color='blue', width=2, dash='dash')),
            
            go.Scatter(x=np.linspace(-3, 2, 100), 
                       y=ols.predict(np.linspace(-3, 2, 100).reshape(-1, 1)), 
                       mode='lines', 
                       name='OLS - Mínimos Cuadrados', 
                       line=dict(color='red', width=2)),
            
            go.Scatter(x=X.flatten(), 
                       y=y, 
                       mode='markers', 
                       name='Datos reales', 
                       marker=dict(color='#C9C9C9', size=8, opacity=0.5))
        ],
        name=f'Frame {i+1}'
    ))

fig = go.Figure(
    frames=frames,
    layout=go.Layout(
        title="Comparación de ajuste: OLS vs SGD",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])],
            
            x=0.8,             
            y=1.17,               
            xanchor="center",  
            yanchor="top")],
        
        legend=dict(
            orientation="h",  
            yanchor="bottom", 
            y=-0.2,  
            xanchor="center",  
            x=0.5  
        ),
        height=600,
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',  
            gridwidth=1  
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',  
            gridwidth=1  
        )
    )
)

fig.add_trace(go.Scatter(x=np.linspace(-3, 2, 100), 
                        y=sgd.predict(np.linspace(-3, 2, 100).reshape(-1, 1)),
                        mode='lines', 
                        name=f'SGD - Iteración {i+1}', 
                        line=dict(color='blue', width=2, dash='dash')))

fig.add_trace(go.Scatter(x=np.linspace(-3, 2, 100), 
                        y=ols.predict(np.linspace(-3, 2, 100).reshape(-1, 1)),
                        mode='lines', 
                        name='OLS - Mínimos Cuadrados',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=X.flatten(), 
                        y=y, 
                        mode='markers', 
                        name='Datos reales', 
                        marker=dict(color='#C9C9C9', size=8, opacity=0.5)))
        
fig.show()
''',language='python')

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

ols = LinearRegression()
ols.fit(X, y)
y_pred_ols = ols.predict(X)

sgd = SGDRegressor(max_iter=1, tol=1e-3, random_state=42, warm_start=True)

frames = []

for i in range(30):
    sgd.partial_fit(X, y)  # Ajusta una iteración
    y_pred_sgd = sgd.predict(X)
    
    frames.append(go.Frame(
        data=[
            # Añadir las líneas de SGD y OLS primero
            go.Scatter(x=np.linspace(-3, 2, 100), 
                       y=sgd.predict(np.linspace(-3, 2, 100).reshape(-1, 1)),
                       mode='lines', 
                       name=f'SGD - Iteración {i+1}', 
                       line=dict(color='blue', width=2, dash='dash')),
            
            go.Scatter(x=np.linspace(-3, 2, 100), 
                       y=ols.predict(np.linspace(-3, 2, 100).reshape(-1, 1)), 
                       mode='lines', 
                       name='OLS - Mínimos Cuadrados', 
                       line=dict(color='red', width=2)),
            
            # Añadir los puntos después de las líneas para que queden detrás
            go.Scatter(x=X.flatten(), 
                       y=y, 
                       mode='markers', 
                       name='Datos reales', 
                       marker=dict(color='#C9C9C9', size=8, opacity=0.5))
        ],
        name=f'Frame {i+1}'
    ))

# Crear la figura
fig = go.Figure(
    frames=frames,
    layout=go.Layout(
        title="Comparación de ajuste: OLS vs SGD",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])],
            
            x=0.8,             # Posición en el eje x (0 es izquierda, 1 es derecha)
            y=1.17,               # Posición en el eje y (0 es abajo, 1 es arriba)
            xanchor="center",  # Ancla para la posición x (puede ser 'left', 'center', 'right')
            yanchor="top")],
        
        legend=dict(
            orientation="h",  # Establecer la orientación de la leyenda a horizontal
            yanchor="bottom",  # Anclar la leyenda al fondo
            y=-0.2,  # Desplazar la leyenda hacia abajo
            xanchor="center",  # Centrar la leyenda
            x=0.5  # Centrar la leyenda horizontalmente
        ),
        height=600,
        xaxis=dict(
            showgrid=True,  # Mostrar la grilla en el eje X
            gridcolor='lightgray',  # Color de la grilla
            gridwidth=1  # Grosor de la línea de la grilla
        ),
        yaxis=dict(
            showgrid=True,  # Mostrar la grilla en el eje Y
            gridcolor='lightgray',  # Color de la grilla
            gridwidth=1  # Grosor de la línea de la grilla
        )
    )
)

# Añadir las líneas de SGD y OLS al gráfico fuera de la animación
fig.add_trace(go.Scatter(x=np.linspace(-3, 2, 100), 
                        y=sgd.predict(np.linspace(-3, 2, 100).reshape(-1, 1)),
                        mode='lines', 
                        name=f'SGD - Iteración {i+1}', 
                        line=dict(color='blue', width=2, dash='dash')))

fig.add_trace(go.Scatter(x=np.linspace(-3, 2, 100), 
                        y=ols.predict(np.linspace(-3, 2, 100).reshape(-1, 1)),
                        mode='lines', 
                        name='OLS - Mínimos Cuadrados',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=X.flatten(), 
                        y=y, 
                        mode='markers', 
                        name='Datos reales', 
                        marker=dict(color='#C9C9C9', size=8, opacity=0.5)))

# Mostrar el gráfico en Streamlit
_, col1, _ = st.columns([0.1, 0.2, 0.1])

with col1:
    st.plotly_chart(fig)


### 4 punto
st.write('''
        <p style='font-size:25px;'><b>
        4. ¿Qué función en R está diseñada para aplicar el descenso del gradiente en la regresión lineal y 
        en qué casos es útil utilizarla?
        </b></p>
        ''', unsafe_allow_html=True)


st.code('''
library(ggplot2)
library(glmnet)

set.seed(123)
n <- 100
x <- rnorm(n)
y <- 3 * x + rnorm(n)

X <- as.matrix(cbind(1, x))
        
# Ajuste con OLS 
ols_model <- lm(y ~ x)

# Ajuste con glmnet
ridge_model <- glmnet(X, y, alpha = 0)  # alpha = 0 para Ridge

lasso_model <- glmnet(X, y, alpha = 1)  # alpha = 1 para Lasso

df <- data.frame(x = x, y = y)

# Predicciones con OLS, Ridge y Lasso
pred_ols <- predict(ols_model, newdata = data.frame(x = x))
pred_ridge <- predict(ridge_model, s = 0, newx = X)  # s = 0 para el valor de lambda = 0
pred_lasso <- predict(lasso_model, s = 0, newx = X)  # s = 0 para el valor de lambda = 0

ggplot(df, aes(x = x, y = y)) +
  geom_point(aes(color = "Datos reales"), alpha = 0.7) +
  geom_line(aes(y = pred_ols, color = "Ajuste OLS"), linetype = "dashed",size = 1) +
  geom_line(aes(y = pred_ridge, color = "Ajuste Ridge"), linetype = "solid",size = 1) +
  geom_line(aes(y = pred_lasso, color = "Ajuste Lasso"), linetype = "dotdash",size = 1) +
  labs(title = "Comparación de ajuste: OLS, Ridge y Lasso",
       color = "Líneas") +
  theme_minimal() +
  scale_color_manual(values = c("Datos reales" = "gray", 
                                "Ajuste OLS" = "blue", 
                                "Ajuste Ridge" = "red", 
                                "Ajuste Lasso" = "green")) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(size = guide_legend(title = "Pesos"))
''')

#process2 = subprocess.Popen(["Rscript", "glmnet.r"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#result2 = process2.communicate()
image2 = Image.open('glmnet.png')

_, col1, _ = st.columns([0.1,0.2,0.1])

with col1:
        st.image(image2,width=600)
        st.caption('''**Figure 1.** Se asignaron pesos mayores a los datos se asignarion si tienen un valor a 
                   5 que son los que corresponden al color rojo''')

### 5 punto
st.write('''
        <p style='font-size:25px;'><b>
        5. ¿Qué significan los parámetros fit intercept, normalize y positive en la clase LinearRegression de Scikitlearn?
        </b></p>
        ''', unsafe_allow_html=True)

st.write('''
        <p style='font-size:23px;'>En Scikit-learn para entrenar un modelo de regresion 
        lineal se hace mediante la clase LinearRegresion del modulo linear_model, esta clase
        tiene los siguientes parametros:</p>''', 
         unsafe_allow_html=True)

st.code(
'''
from sklearn.linear_model import LinearRegression # Existe un modulo unicamente para modelos lineales :0

model = LinearRegression(

        fit_intercept: bool, default=True 
        #Parametro para calcular el intercepto o no. Si se establece en False, no se utilizará ningún intercepto 
        #en los cálculos pero se espera que los datos estén centrados). 

        copy_X: bool, default=True
        #Parametro para hacer una copia de los datos, Si se marca como False se sobreescribiran los cambios en los datos originales 

        positive: bool, default=False
        #Cuando se establece en True, obliga a que los coeficientes sean positivos. Esta opción sólo es compatible con matrices densas.

        n_jobs: int, default=None
        #número de núcleos que el algoritmo debe utilizar para procesar la tarea. Esto sólo proporcionará un aumento de velocidad en el 
        #caso de problemas suficientemente grandes, esta opcion se positive se establece en True. Si el parametro es None significa 1 a 
        #menos que esté en un contexto joblib.parallel_backend. -1 significa utilizar todos los procesadores.

        #Si el parámetro positive es True, añade una restricción adicional en el ajuste del modelo, y el cálculo se vuelve más complejo 
        #porque el algoritmo tiene que asegurarse de que todas las soluciones cumplen con esta restricción, por lo que no puede usar el 
        #método estándar de mínimos cuadrados ordinarios. En su lugar, necesita utilizar un algoritmo de optimización restringida, como 
        #Least Squares with Non-Negative Constraints (NNLS), que tiene un costo computacional mayor y por lo tanto se necesitan mas jobs.
        )
''',language='python'
)

### 6 punto
st.write('''<p style='font-size:25px;'><b>
        6. ¿Qué significan los parámetros formula, data, subset, weights, na.action, y method en la función lm() de R?
        <b></p>''', unsafe_allow_html=True)

st.write("<p style='font-size:23px;'>En R para entrenar un modelo de regresion \
         lineal se hace mediante la funcion lm(), esta funcion \
         tiene los siguientes parametros:</p>", 
         unsafe_allow_html=True)

st.code('''

model <- lm(
        formula 
        #Descripción simbólica del modelo, algo como "Y ~ X1 + X2"
        
        data 
        #array o dataframe que contiene las variables del modelo
        
        subset 
        #Vector opcional que especifica un subconjunto de observaciones a usar en el ajuste.

        weights
        #Vector opcional de pesos para el ajuste,

        na.action
        #Función que indica qué hacer con los datos que contienen NAs. Puede ser NULL, na.omit, o na.exclude.

        method: default "qr"
        #Método a utilizar, solo se admite method = "qr".

        model: default TRUE 
        #Indica si quiero retornar los atributos del modelo

        x: default FALSE
        #Indican si se debe devolver la matriz del modelo

        y: default FALSE
        #Indican si se debe devolver la variable dependiente

        qr: default TRUE
        #Indican si se debe devolver la descomposición QR

        singular.ok: default TRUE 
        #Indica si un ajuste singular debe ser considerado como un error.

        contrasts: default NULL 
        #Controla cómo maneja las variables categóricas en el modelo. Cuando se tienen factores (variables categóricas) en el modelo, 
        #necesita codificarlos numéricamente antes de realizar el ajuste, la forma en que se codifican estos factores se conoce como contraste.      

        offset: default NULL
        #Permite incluir un componente conocido en el predictor lineal del modelo. Este valor se resta del vector de respuesta antes de calcular 
        #los coeficientes del modelo, lo que significa que no se estima un coeficiente para offset; simplemente se ajusta el modelo con este 
        #término como parte de la ecuación.
        )
''', language="python")

st.write('''<p style='font-size:23px;'>
        El modelo no necesita de indicar un entrenamiento, al hacer model <- lm(...) y ajustar lo que queramos,
        el modelo se entrena de inmediato y usa la tecnica de Minimos cuadrados si no se indicaron pesos, si se indicaron pesos se usa minimos
        cuadrados ponderados con los respectivos
        </p>''', 
        unsafe_allow_html=True)

### 7 punto
st.write('''<p style='font-size:25px;'><b>
        7. ¿Qué significan los parámetros alpha, lambda, standardize, y family en la función glmnet de R?
        <b></p>
        ''', unsafe_allow_html=True)

st.code('''
glmnet(
        x
        #Es la matriz de variables independientes, requisito: Debe tener al menos dos columnas (es decir, nvars > 1).

        y
        #Es la variable de respuesta. Puede ser de distintos tipos dependiendo del modelo

        family
        #Especifica el tipo de modelo o distribución de probabilidad para los datos de salida y. Cada familia tiene su propio 
        #tipo de modelo adecuado para ciertos tipos de datos:

        #gaussian: Regresión lineal estándar, adecuada para datos continuos.

        #binomial: Regresión logística binaria para datos de respuesta binaria. Permite realizar clasificación con salida de 0 y 1.
        
        #poisson: Modelo para datos de conteo (no negativos), como el número de eventos que ocurren en un intervalo de tiempo.
        
        #multinomial: Regresión multinomial para clasificación en múltiples categorías, donde y es una variable categórica con más de dos niveles.
        
        #cox: Modelo de Cox para análisis de supervivencia, donde y es un objeto Surv (paquete survival en R).
        
        #mgaussian: Para modelos multivariados de respuesta continua, donde y es una matriz de varias variables de respuesta.
        
        weights: default NULL
        # Vector de pesos para cada observación. Por defecto es 1 para cada observación.

        offset: default NULL
        #Permite incluir un componente conocido en el predictor lineal del modelo. Este valor se resta del vector de respuesta antes de calcular 
        #los coeficientes del modelo, lo que significa que no se estima un coeficiente para offset; simplemente se ajusta el modelo con este 
        #término como parte de la ecuación.

        alpha: default 1
        #Controla el tipo de regularización en el modelo, combinando la penalización de Ridge y Lasso según el valor que adopte entre 0 y 1.
        
        #alpha = 1: Produce la penalización Lasso, que aplica regularización L1, forzando algunos coeficientes a ser exactamente cero y 
        #favoreciendo la selección de variables.

        #alpha = 0: Produce la penalización Ridge, que aplica regularización L2. Ridge penaliza la magnitud de los coeficientes sin forzar 
        #a ninguno a ser exactamente cero, ayudando a reducir la varianza del modelo cuando las variables son colineales.

        #0 < alpha < 1: Se obtiene una combinación de ambas penalizaciones, conocida como Elastic Net, que permite un balance entre selección 
        #de variables (Lasso) y estabilidad frente a colinealidad (Ridge). Elastic Net es útil cuando hay muchas variables correlacionadas, 
        #ya que tiende a agrupar estas variables en el modelo en lugar de elegir solo una.

        nlambda: default 100
        #Número de valores de lambda a probar

        lambda.min.ratio: default ifelse(nobs < nvars, 0.01, 1e-04)
        #El valor más pequeño de lambda como una fracción de lambda.max

        lambda: default NULL
        #Secuencia de valores de lambda suministrados por el usuario. Si no se proporciona, el algoritmo genera su propia secuencia.

        standardize: default TRUE

        #Este parámetro determina si las variables predictoras x deben ser estandarizadas (escaladas para tener media cero y varianza uno) antes 
        #de ajustar el modelo. Esto es importante en modelos de regularización, ya que sin estandarización, las variables con escalas más grandes 
        #pueden dominar la penalización, produciendo coeficientes sospechosos

        intercept: default TRUE
        #Parametro para calcular el intercepto o no.

        thresh: default 1e-07
        #Umbral de convergencia para el descenso por coordenadas. Cada iteración continúa hasta que el cambio máximo en el objetivo después 
        #de cualquier actualización de coeficientes sea menor que este umbral.        

        dfmax: default nvars + 1
        #Limita el número máximo de variables en el modelo. Esto puede ser útil para conjuntos de datos con muchas variables.

        pmax: default min(dfmax * 2 + 20, nvars)
        #Limita el número máximo de variables que pueden ser no cero en el modelo.

        exclude: default NULL
        #Índices de las variables que deben ser excluidas del modelo. Esto es equivalente a aplicar un factor 
        #de penalización infinito a esas variables.        

        penalty.factor: default rep(1, nvars)
        #Factores de penalización separados que se pueden aplicar a cada coeficiente. Esto permite diferenciar el decrecimiento de las variables. 
        #Puede ser 0 para algunas variables, lo que significa que no se aplicará penalización a esas variables.

        lower.limits: default -Inf
        #Límites inferiores para los coeficientes.

        upper.limits: default Inf
        #Límites superiores para los coeficientes.

        maxit: default 1e+05
        Número máximo de pasadas sobre los datos para cada valor de lambda

        type.gaussian: default ifelse(nvars < 500, "covariance", "naive")
        #Tipo de algoritmo para la familia "gaussian". "covariance" es más rápido para un número pequeño de variables, 
        #mientras que "naive" es más eficiente cuando hay muchas más variables que observaciones.

        type.logistic: default c("Newton", "modified.Newton")
        #Si se debe usar el algoritmo de Newton para ajustar un modelo logístico.
        )
''',language="python")

### 8 punto
st.write('''<p style='font-size:25px;'><b>
        8. ¿Qué significan los parámetros loss, penalty, alpha, y max iter en la clase SGDRegressor de Scikit-learn?
        </b></p>''', unsafe_allow_html=True)

st.code('''
SGDRegressor(

        loss: str, default=squared_error
        #La función de pérdida a utilizar. Puede ser squared_error para mínimos cuadrados,
        #huber para reducir el impacto de outliers, epsilon_insensitive que ignora errores menores a epsilon,
        #o squared_epsilon_insensitive que combina ambas.

        epsilon: float, default=0.1
        #Umbral para las funciones de pérdida huber y epsilon_insensitive.
            
        penalty: {l2, l1, elasticnet, None}, default=l2
        #El término de regularización. None desactiva la penalización.

        alpha: float, default=0.0001
        #Constante que multiplica el término de regularización.

        l1_ratio: float, default=0.15
        #Parámetro de mezcla para Elastic Net; controla la proporción de L1 y L2.

        fit_intercept: bool, default=True
        #Indica si se debe estimar el intercepto. Si es False, se asume que los datos están centrados.

        max_iter: int, default=1000
        #Máximo número de épocas sobre los datos de entrenamiento.

        tol: float or None, default=1e-3
        #Criterio de parada. Si es None, el entrenamiento se detiene cuando no mejora después de n_iter_no_change.

        shuffle: bool, default=True
        #Indica si los datos de entrenamiento deben mezclarse después de cada época.

        verbose: int, default=0
        #Nivel de detalle del proceso. 

        random_state: int or RandomState, default=None
        #Controla la aleatoriedad en el barajado de datos cuando shuffle es True.

        learning_rate: str, default='invscaling'
        #Estrategia de ajuste de tasa de aprendizaje; puede ser 'constant', 'optimal', 'invscaling', o 'adaptive'.

        eta0: float, default=0.01
        #Tasa de aprendizaje inicial para los ajustes 'constant', 'invscaling' y 'adaptive'.

        power_t: float, default=0.25
        #Exponente para la tasa de aprendizaje de escalado inverso.

        early_stopping: bool, default=False
        #Si True, se usa para detener el entrenamiento cuando el rendimiento de validación no mejora.

        validation_fraction: float, default=0.1
        #Proporción de datos de entrenamiento usados como conjunto de validación.

        n_iter_no_change: int, default=5
        #Número de iteraciones sin mejora antes de detener el entrenamiento.

        warm_start: bool, default=False
        #Si es True, reutiliza la solución previa al llamar a fit nuevamente.

        average: bool or int, default=False
        #Si es True, calcula el promedio de los pesos actualizados en cada paso.

)''',language="python")