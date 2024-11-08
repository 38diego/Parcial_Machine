import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

st.write('''
# Predictoras
**crim**
per capita crime rate by town.
    
zn
proportion of residential land zoned for lots over 25,000 sq.ft.

indus
proportion of non-retail business acres per town.

chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox
nitrogen oxides concentration (parts per 10 million).

rm
average number of rooms per dwelling.

age
proportion of owner-occupied units built prior to 1940.

dis
weighted mean of distances to five Boston employment centres.

rad
index of accessibility to radial highways.

tax
full-value property-tax rate per \$10,000.

ptratio
pupil-teacher ratio by town.

black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat
lower status of the population (percent).


# Obejivo 
**medv**
median value of owner-occupied homes in \$1000s.''')

train_data = pd.read_csv("train_data.csv")
test_data = pd.DataFrame("test_data.csv")

train_data.to_csv("train_data.csv",index=False)
test_data.to_csv("test_data.csv",index=False)

X_train_sm = sm.add_constant(train_data)
model_sm = sm.OLS(train_targets, X_train_sm).fit()

# Predict on the training set to calculate residuals
y_train_pred_sm = model_sm.predict(X_train_sm)
residuals_sm = model_sm.resid

st.write(model_sm.summary())

###(a) Supuesto de media cero: Graficar los residuos ei contra los valores predichos ˆyi y comprobar si los residuos
###est´an distribuidos aleatoriamente alrededor de la l´ınea de cero.
###(b) Supuesto de homocedasticidad: Verificar si los residuos muestran una varianza constante a lo largo de
###los valores predichos, es decir, si no forman patrones visibles como conos o par´abolas.
###(c) Supuesto de independencia: Graficar los residuos en funci´on del orden de las observaciones para comprobar si hay dependencia entre ellos.

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred_sm, y=residuals_sm)
plt.axhline(y=np.mean(residuals_sm), color='r', linestyle='--', label='Media residuos')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted values")
plt.legend()
st.pyplot(plt)

#Supuesto de normalidad: Realizar un gr´afico Q-Q (quantile-quantile) para comparar la distribuci´on de
#los residuos con una distribuci´on normal te´orica.

plt.figure(figsize=(10, 6))
sm.qqplot(residuals_sm, line='s',)
plt.title("Q-Q plot of residuals")
st.pyplot(plt)


#Supuesto de ausencia de multicolinealidad: Calcular el factor de inflaci´on de la varianza (VIF) para
#cada variable predictora y verificar si alg´un valor es superior a 10, lo cual indicar´ıa multicolinealidad.

vif_data = pd.DataFrame()
vif_data["Feature"] = train_data.columns
vif_data["VIF"] = [variance_inflation_factor(train_data.values, i) for i in range(train_data.shape[1])]
st.dataframe(vif_data)


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