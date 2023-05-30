#!/usr/bin/env python
# coding: utf-8

# # Este archivo es el código base para la parte práctica del examen del tercer parcial de la clase de Inteligencia Artificial
# 
# ## Instrucciones
# 
# Por favor lea cuidadosamente y codifique lo que se le pida. Las instrucciones se verán de la siguiente manera:
# ```
# ***# Este ese el formato para las instrucciones***
# ```
# 
# ## Entregables
# 
# Se entrega individualmente un archivo comprimido en Zip que contenga:
# 
# *   La libreta de Colab (extensión ipynb)
# *   Ls libreta de Colab en formato PDF; use la opción de imprimir para generarlo
# 
# ## Evaluación
# 
# *   La celda requerida debe ejecutar sin errores: 1 pts
# *   La celda requerida implementa el código solicitado: 1pts
# 
# ## Código de ética profesional
# 
# Al entregar este archivo con sus implementaciones, acepta que el trabajo realizado es de su autoría y que de confirmarse lo contrario se anulará su examen.
# 
# Recuerde, el resultado de un trabajo por mérito propio siempre es satisfactorio

# 
# ```
# Haga doble clic para editar la celda y llenar los datos correspondientes:
# ```
# 
# Nombre del estudiante:
# 
# Fecha de entrega:
# 
# 
# 

# In[40]:


### import the libraries and modules required

# libraries to manipulate the data and to visualise it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
# this is the library that contains the NN capabilities
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# the evaluation metrics for classification
from sklearn.metrics import accuracy_score
# for hyper parameter tuning
from sklearn.model_selection import GridSearchCV
# to meassure the execution time of the neural networks
import time


# ## Carga del conjunto de datos

# `Cargue la base de datos gym-1sec.csv y muestre los primeros 5 registos:`

# In[41]:


dataset = pd.read_csv(filepath_or_buffer='gym-1sec.csv', sep=',')

print('Dataset size {} columns and {} rows'.format(dataset.shape[1], dataset.shape[0]))

dataset.head()


# ```
# Visualice la cantidad de datos que hay en cada una de los differentes valores de la característica (feature) nivel de occupación.
# 
# La descripción de las variables es como sigue de izquierda a derecha:
# 
# *   Fecha de registro
# *   Presión barométrica en hecto-pascal
# *   Altura relativa desde el nivel del mar en metros
# *   Humedad relativa en porcentaje
# *   Temperatura en grados celcius
# *   Nivel de ocupación en etiquetas
# ```
# 
# 

# In[42]:


dataset.groupby('occ').size()


# ```
# Utilice el análisis de los 5 números para visualizar la descripción del cconjunto de datos
# ```
# 
# 

# In[43]:


dataset.describe()


# ## Visualización de los datos

# ```
# Modifique el siguiente código para visualizar la humedad del gymnasio de todos los niveles de ocupación (en el eje y) respecto a la altura (en el eje x)
# 
# Nombre la gráfiica adecuadamente a los datos que se muestran
# ```
# 
# 

# In[44]:


y = dataset['hum']
x = dataset['alt']

fig, ax = plt.subplots()
fig.set_size_inches(10, 5)

ax.scatter(x, y, alpha=0.5)

ax.set_xlabel('Altura')
ax.set_ylabel('Humedad')
ax.set_title('Humedad vs Altura')
plt.show()


# ## Generación del conjunto de datos de entrenamiento y testeo

# ```
# Genere los conjuntos de datos de entrenamiento y testeo de la siguiente manera:
# 
# *   El conjunto de testeo debe ser el 5% aleatorio del conjunto de datos total
# *   Para el conjunto de entrenamiento, utilice todas las variables/características del conjunto de datos con excepción de:
#  'date','occ'
# *   Seleccione la varible de nivel de ocupación como la variable/característica de interés para hacer la clasificación
# ```
# 
# 

# In[45]:


X = dataset.drop(['date','occ'], axis=1)
Y = dataset['occ']

# IF N, Convert to numpy array
X = np.array(X)
Y = np.array(Y)

# # VALIDATE SHAPES
# print('X shape: {}'.format(X.shape))
# print('Y shape: {}'.format(Y.shape))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)


# ## Escalamiento

# ```
# Si es necesario escalar los datos hágalo; en caso contrario, escribir no es necesario escalar los datos en la celda como comentario
# ```
# 
# 

# In[46]:


print('Min values of the dataset are: \n{}'.format(dataset.min()))
print('Max values of the dataset are: \n{}'.format(dataset.max()))

# Determine if the data needs to be scaled
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print('Min values of the dataset are: \n{}'.format(X_train_scaled.min(axis=0)))
# print('Max values of the dataset are: \n{}'.format(X_train_scaled.max(axis=0)))

print('Min values of the dataset are: \n{}'.format(X_test_scaled.min()))
print('Max values of the dataset are: \n{}'.format(X_test_scaled.max()))


# ## Clasificación

# ### Definición del modelo

# ```
# Defina un modelo de red neuronal con las siguientes características:
# 
# *   4 capas en total
# *   La capa de salida debe tener el mimso número de neuronas que tipos/clases de nivel de ocupación
# *   La capa de entrada debe tener el mismo número de neuroas que tipos/clases de nivel de ocupación
# *   Las capas intermedias debe tener +2 y +1 del número de neuronas de entrada menos el número de características/variables eliminadas del conjunto de entrenamiento
# *   Un máximo número de iteraciones del número de clases/tipos de nivel de ocupación multiplicado por 100
# *   Usar la función de activación tangente hyperbólica
# *   User el solucionador descenso de gradiente estocástico
# ```
# 
# 

# In[47]:


mlp_clf = MLPClassifier(hidden_layer_sizes=(Y_train.shape[0]+2, Y_train.shape[0]+1), activation='tanh', solver='sgd', max_iter=Y_train.shape[0]*100, random_state=42)


# ### Entrenamiento
# 
# ```
# Entrene el modelo definido con anterioridad y muestrer su tiempo de ejecución en segundos
# ```
# **Importante:** esto puede tomar aproximadamente unos 5 minutos
# 
# 

# In[ ]:


start_time = time.time()
mlp_clf.fit(X_train_scaled, Y_train)
end_time = time.time()

print('Elapsed time: {} seconds'.format(end_time - start_time))


# ### Testeo y evaluación del modelo
# 
# ```
# Utilice el modelo entrenado para hacer la clasificación de los datos de testeo
# ```
# 
# 

# In[ ]:


from sklearn.metrics import classification_report

y_pred = mlp_clf.predict(X_test_scaled)
y_pred

print(classification_report(Y_test, y_pred))


# 
# ```
# Evalue el modelo de clasificación entrenado previamente para mostrar su accuracy
# ```
# 

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy: {}'.format(accuracy_score(Y_test, y_pred)))


# 
# ```
# Visualice la curva de entrenamiento
# ```
# 
# 

# In[ ]:


plt.plot(mlp_clf.loss_curve_)
plt.title('Loss Curve', fontsize=12)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# 
# ```
# Visualice la matriz de confusión 
# ```
# 
# 

# In[ ]:


confusion_matrix = pd.crosstab(Y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()


# 
# ```
# Con el valor del accuracy y las gráficas de entrenamiento y matriz de confusión explique brevemente el rendimiento del modelo de red neuronal entrenado
# ```
# 
# 

# **Doble clic aquí para editar su respuesta**

# ### Búsqueda de hiperparámetros a través de una matriz
# 
# 
# 
# ```
# Defina una matriz de búsqueda de hiperparámetros que contenga lo siguiente:
# 
# *   Un modelo de red neuronal con el número de capas y neuronas que considere lleve a mejorar el aprendizaje en la tarea de clasificación
# *   El número de iteraciones que consideren sean pertinentes
# *   El optimizador adam
# *   La función de activación hyperbolic relu
# ```
# 
# 

# In[ ]:


param_grid = {
    'hidden_layer_sizes': [(Y_train.shape[0]+2, Y_train.shape[0]+1), (Y_train.shape[0]+3, Y_train.shape[0]+2), (Y_train.shape[0]+4, Y_train.shape[0]+3)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'max_iter': [Y_train.shape[0]*100, Y_train.shape[0]*200, Y_train.shape[0]*300]
}


# 
# ```
# Defina la búsqueda de la matriz sin validación cruzada
# ```
# 
# 

# In[ ]:


grid = GridSearchCV(mlp_clf, param_grid, n_jobs=-1)


# 
# ```
# Execute la búsqueda de los hiperparámetros y muestre el tiempo total de execución
# ```
# 
# **Importante:** tenga en mente que esta sección puede tardar varios minutos, de acuerdo a los modelos incluídos en la búsqueda

# In[ ]:


start_time = time.time()
grid.fit(X_train_scaled, Y_train)
end_time = time.time()

print('Elapsed time: {} seconds'.format(end_time - start_time))


# 
# ```
# Muestre el 'mejor modelo' de la búsqueda de los hiperparámetros
# ```
# 
# 

# In[ ]:


print('The best hyper parameter values are:\n{}'.format(grid.best_params_))
grid_results = pd.DataFrame(grid.cv_results_)


# 
# ```
# Obtenga y muestre las predicciones con el mejor modelo de la búsqueda de hiperparámetros
# ```
# 
# 

# In[ ]:


print('Best model predictions: {}'.format(grid.best_estimator_.predict(X_test_scaled)))
grid_results.sort_values(by='rank_test_score', limit=1)

print('RANK')
grid_results.sort_values(by='rank_test_score')


# 
# ```
# Evalue el mejor modelo de clasificación resultante de la búsqueda de hiperparámetros para mostrar su accuracy
# ```
# 
# 

# In[ ]:


best_model = MLPClassifier(hidden_layer_sizes=grid.best_params_['hidden_layer_sizes'], activation=grid.best_params_['activation'], solver=grid.best_params_['solver'], max_iter=grid.best_params_['max_iter'], random_state=42)
best_model.fit(X_train_scaled, Y_train)

y_pred = best_model.predict(X_test_scaled)
print('Accuracy: {}'.format(accuracy_score(Y_test, y_pred)))


# ```
# Visualice la matriz de confusión del mejor modelo de clasificación resultante de la búsqueda de hiperparámetros
# ```
# 
# 

# In[ ]:


best_mode_confusion_matrix = pd.crosstab(Y_test, grid.best_estimator_.predict(X_test_scaled), rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(best_mode_confusion_matrix, annot=True)
plt.show()


# ## Conclusiones
# 
# ```
# De acuerdo a los resultados de las métricas de evaluación y a las gráficas de matriz de confusión, mencione qué modelo de red neuronal se desempeñó mejor y por qué considera que fue así
# ```
# 
# 

# **Doble clic aquí para editar su respuesta**
# 
# 
