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

st.write("<p style='font-size:23px;'>En Scikit-learn para entrenar un modelo de regresion \
         lineal se hace mediante la clase LinearRegresion del modulo linear_model, esta clase \
         tiene los siguientes parametros:</p>", 
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
        Si este parametro tiene pesos, usa la tecnica de minimos cuadrados ponderados con los pesos indicados en este parametro.
        </p>''', unsafe_allow_html=True)

process1 = subprocess.Popen(["Rscript", "helloworld.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
result1 = process1.communicate()
image = Image.open('plot.png')
st.image(image)


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
'''