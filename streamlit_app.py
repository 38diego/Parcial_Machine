import streamlit as st
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from streamlit_navigation_bar import st_navbar
from PIL import Image

st.set_page_config(layout="wide")

st.markdown('# :red[Métodos y parámetros en regresión lineal]')

### 1 punto
st.write("<p style='font-size:25px;'>1. ¿Qué método utiliza la clase LinearRegression de \
         Scikit-learn por defecto para ajustar un modelo de regresión lineal?</p>", 
         unsafe_allow_html=True)

st.code('''
model.fit(
        
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
        #Un arreglo que contiene los datos del conjunto de entrenamiento.

        y: array-like of shape (n_samples,) or (n_samples, n_targets)
        #Un arreglo que contiene la(s) variable(s) objetivo.

        sample_weight: array-like of shape (n_samples,), default=None
        #Un arreglo que especifica los pesos para cada distintas muestras de los datos.
)

#Algunos atributos de esta clase son:

model.coef_array -> shape (n_features, ) or (n_targets, n_features)
#Este atributo almacena los coeficientes estimados de la regresión lineal, es decir, los 
#valores de las pendientes para cada característica (variable independiente) en el modelo ajustado.

>>> array([1., 2.]) ### Solo los coeficientes, no el intercepto

model.intercept_ -> float or array of shape (n_targets,)
#Este es el término independiente o intercepto del modelo, Si fit_intercept = False, este valor se establece en 0.0

>>> 3.

model.singular_ -> array of shape (min(X, y),)
#Los valores singulares se pueden usar para detectar problemas de colinealidad entre las características. Si algunos de estos valores son 
#cercanos a cero, puede indicar que una o más características son casi linealmente dependientes.

>>> array([1.61803399, 0.61803399])


model.n_features_in_ -> int
#Indica el número de características o variables independientes que el modelo ha visto durante el ajuste 
 
>>> 2
        
model.rank_ -> int
#Este atributo representa el rango de la matriz de características X, que es una medida de la independencia lineal de las columnas de X

>>> 2 #Si este numero es menor a el atributo .n_features_in_, indica multicolinealidad        
''',language = "python")


### 2 punto
st.write('''<p style='font-size:25px;'>
        2. En R, cuando utilizamos la función lm() para ajustar un modelo de regresión lineal,
        ¿qué técnica se emplea paracalcular los coeficientes?
        </p>''', unsafe_allow_html=True)

st.markdown('''
        <p style='font-size:23px;'>
        En R al usar lm() usa la tecnica de minimos cuadrados ordinarios cuando no se indican pesos en el parametro weights, 
        Si este parametro tiene pesos, usa la tecnica de minimos cuadrados ponderados con los pesos indicados en este parametro, 
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

# Asignar pesos más extremos
weights <- ifelse(x > 5, 100, 0.01)  # Pesos muy altos para x > 5, muy bajos para el resto

# Ajustar el modelo sin ponderación
modelo_sin_ponderacion <- lm(y ~ x)

# Ajustar el modelo con ponderación
modelo_con_ponderacion <- lm(y ~ x, weights = weights)

# Crear un data frame con las predicciones de ambos modelos
datos <- data.frame(x = x, y = y, weights = weights)
datos$pred_sin_ponderacion <- predict(modelo_sin_ponderacion)
datos$pred_con_ponderacion <- predict(modelo_con_ponderacion)

# Crear una nueva variable para resaltar los puntos con pesos altos
datos$highlight <- ifelse(datos$weights == 100, "Peso Alto", "Peso Bajo")

# Graficar los datos y las dos líneas de regresión
ggplot(datos, aes(x = x, y = y)) +
  geom_point(aes(color = highlight), alpha = 0.7) +  # Colorear según el tipo de peso
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
  ) +  # Colorear las líneas y los puntos
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(size = guide_legend(title = "Pesos"))     
''')

process1 = subprocess.Popen(["Rscript", "helloworld.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
result1 = process1.communicate()
image = Image.open('plot.png')

_, col1, _ = st.columns([0.1,0.2,0.1])

with col1:
        st.image(image,width=600)
        st.caption('''**Figure 1.** Se asignaron pesos mayores a los datos se asignarion si tienen un valor a 
                   5 que son los que corresponden al color rojo''')

st.write('''<p style='font-size:25px;'>
        3. ¿Qué método está diseñado en Scikit-learn para aplicar el descenso del gradiente en la regresión lineal y 
        cuándo es útil utilizarlo?</p>''', 
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

# Modelo de Descenso del Gradiente (SGD)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X, y)

# Predicciones de Descenso del Gradiente
y_pred_sgd = sgd.predict(X)

# Ordenar los datos para visualización
sorted_indices = np.argsort(X.flatten())
X_sorted = X[sorted_indices]
y_pred_ols_sorted = y_pred_ols[sorted_indices]
y_pred_sgd_sorted = y_pred_sgd[sorted_indices]

# Visualizar los resultados
plt.figure(figsize=(8, 8))
plt.scatter(X, y, color='#C9C9C9', label='Datos reales')
plt.plot(X_sorted, y_pred_ols_sorted, color='red', label='OLS - Mínimos Cuadrados', linestyle='-', linewidth=2)
plt.plot(X_sorted, y_pred_sgd_sorted, color='blue', label='SGD - Descenso del Gradiente', linestyle='--', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.title('Comparación de Regresión: Mínimos Cuadrados vs. Descenso del Gradiente')
plt.legend()
plt.show()
''',language='python')

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

# Modelo de Descenso del Gradiente (SGD)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X, y)

# Predicciones de Descenso del Gradiente
y_pred_sgd = sgd.predict(X)

# Ordenar los datos para visualización
sorted_indices = np.argsort(X.flatten())
X_sorted = X[sorted_indices]
y_pred_ols_sorted = y_pred_ols[sorted_indices]
y_pred_sgd_sorted = y_pred_sgd[sorted_indices]

# Visualizar los resultados
plt.figure(figsize=(10, 8))
plt.scatter(X, y, color='#C9C9C9', label='Datos reales')
plt.plot(X_sorted, y_pred_ols_sorted, color='red', label='OLS - Mínimos Cuadrados', linestyle='-', linewidth=2)
plt.plot(X_sorted, y_pred_sgd_sorted, color='blue', label='SGD - Descenso del Gradiente', linestyle='--', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.title('Comparación de Regresión: Mínimos Cuadrados vs. Descenso del Gradiente')
plt.legend()

_, col1, _ = st.columns([0.1,0.2,0.1])

with col1:
        st.pyplot(plt)

### 4 punto

st.write('''
        <p style='font-size:25px;'>
        4. ¿Qué función en R está diseñada para aplicar el descenso del gradiente en la regresión lineal y 
        en qué casos es útil utilizarla?</p>
        ''', unsafe_allow_html=True)


### 5 punto
st.write('''
        <p style='font-size:25px;'>
        5. ¿Qué significan los parámetros fit intercept, normalize y positive en la clase LinearRegression de Scikitlearn?</p>
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
st.write("<p style='font-size:25px;'>6. ¿Qué significan los parámetros formula, data, subset, weights, na.action, y method en la \
        función lm() de R?</p>", unsafe_allow_html=True)

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

        method = "qr"
        #Método a utilizar, solo se admite method = "qr".

        model = TRUE 
        #Indica si quiero retornar los atributos del modelo

        x = FALSE
        #Indican si se debe devolver la matriz del modelo

        y = FALSE
        #Indican si se debe devolver la variable dependiente

        qr= TRUE
        #Indican si se debe devolver la descomposición QR

        singular.ok = TRUE 
        #Indica si un ajuste singular debe ser considerado como un error.

        contrasts = NULL 
        #Controla cómo maneja las variables categóricas en el modelo. Cuando se tienen factores (variables categóricas) en el modelo, 
        #necesita codificarlos numéricamente antes de realizar el ajuste, la forma en que se codifican estos factores se conoce como contraste.      

        offset
        #Permite incluir un componente conocido en el predictor lineal del modelo. Este valor se resta del vector de respuesta antes de calcular 
        #los coeficientes del modelo, lo que significa que no se estima un coeficiente para offset; simplemente se ajusta el modelo con este 
        #término como parte de la ecuación.
        )
''', language='R')

st.write("<p style='font-size:23px;'>El modelo no necesita de indicar un entrenamiento, al hacer model <- lm(...) y ajustar lo que queramos,\
        el modelo se entrena de inmediato y usa la tecnica de Minimos cuadrados si no se indicaron pesos, si se indicaron pesos se usa minimos \
        cuadrados ponderados con los respectivos</p>", 
        unsafe_allow_html=True)

### 7 punto
st.write('''
        7. ¿Qué significan los parámetros alpha, lambda, standardize, y family en la función glmnet de R?
        ''', unsafe_allow_html=True)



### 8 punto
st.write('''<p style='font-size:25px;'>
        8. ¿Qué significan los parámetros loss, penalty, alpha, y max iter en la clase SGDRegressor de Scikit-learn?
        </p>''', unsafe_allow_html=True)

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