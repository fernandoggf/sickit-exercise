# ---------------------------------------------------------------------------------------------------#
# En este ejercicio annalizaremos un dataset de los parámetros
# fisicoquímicos de vinos, con una respuesta de calidad, con la finalidad
# de utilizar modelos de la librerìa de sklearn de Machine Learning
# ---------------------------------------------------------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.multivariate.manova import MANOVA

# Leemos el dataset
wine = pd.read_csv('winequality-red.csv')
# Vemos caracterìsticas del set
print(wine.info())
# De la respuesta vemos que el set esta muy completo
# para verificarlo, analizamos si hay nulls
print(wine.isnull().sum())
# Segumos entendiendo el set
print(wine.nunique())
# Tenemos una gradaciòn de calidad discreta de 6 valores, de 3 a 8
print(wine['quality'].value_counts())
plt.figure(0)
plt.hist(wine['quality'])
plt.savefig('quality_hist.png')
print(wine['quality'].describe())
# de acuerdo al histograma, la mayor ocurrencia de calidad esta en el valor 5 y 6
# Ahora, usaremos la selección de modelo de la libreria sickit para hacer el train, test y split. 
# Usaremos el frame completo, dejando fuera -quality- ya que esta será nuestra variable respuesta (Y).
X = wine.drop('quality', axis=1)
y = wine['quality']
# A continuación, generaremos, a partir de toda nuestra información, dos subsets, uno de entrenamiento y uno de testeo, 
# la proporción es de 20% para el testeo y 80% para el entreno (test_size = 0.2) y añadiendo un valor semilla aleatorio 
# (random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
# Debido a la heterogeneidad de los datos por cada columna, por ejemplo densidad va de 0 a 1 mientras que el dióxido 
# de sulfuro va de 0 a 100, la ponderación de los datos puede ser sesgada, por lo que se van a escalar en proporción a 
# su medida para mantener un parámetro rígido en cuanto a magnitud.
# Generamos el escalamiento
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Solo se realiza un fit_transform ya que el test no lo queremos volver a ajustar, solo transformar
print(X_train[-1], X_test[-1])
# Asi, vemos ahora un escalamiento de todas las columnas, teniendo parámetros más cercanos entre sí.

# ---------------------------------------------------------------------------------------------------#
#Primer metodología: Random Forest Classifier (RFC)
# Este modelo de clasificación basado en Random Forests tiene la característica que es el método con 
# menos partes movibles para afinar, por lo que lo vuelve fácil de usar, sin embargo, esta característica 
# no permite mucha holgura en la manipulación de los datos.
RFC = RandomForestClassifier(n_estimators = 300)
# n_estimators Es el único valor que se manipula y es la cantidad de árboles de desición que queremos en nuestro forest
# Colocamos el set de entrenamiento el modelo RFC y hacemos la predicción con el set de testeo
RFC.fit(X_train, y_train)
prediction_RFC = RFC.predict(X_test)
# Vemos algunas predicciones
prediction_RFC[:10]
# Ahora veamos el comportamiento del modelo
print(classification_report(y_test, prediction_RFC))
''' 
Respuesta: 
   precision    recall  f1-score   support
           3       0.00      0.00      0.00         1
           4       0.00      0.00      0.00        10
           5       0.70      0.75      0.72       130
           6       0.62      0.69      0.65       132
           7       0.62      0.50      0.55        42
           8       0.00      0.00      0.00         5

    accuracy                           0.65       320
   macro avg       0.32      0.32      0.32       320
weighted avg       0.62      0.65      0.64       320
'''
print(confusion_matrix(y_test, prediction_RFC))
'''                  - Quality predecido
[[ 0  0  1  0  0  0] - 3
 [ 0  0  6  4  0  0] - 4
 [ 0  0 97 32  1  0] - 5 ~
 [ 0  0 33 93  6  0] - 6 X
 [ 0  0  0 20 21  1] - 7 
 [ 0  0  0  1  4  0]] - 8'''
# Como vemos en nuestro reporte anterior, la clasificación a partir del test 
# predictivo tiene una precisión de ~0.60, no muy bueno, por lo que intentaremos 
# hacer los mismo con nuestro subset ponderado, dejando fuera los de calidad menor a 6.
wine_qa = wine[wine['quality'] >= 6]
X = wine_qa.drop('quality', axis=1)
y = wine_qa['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
RFC = RandomForestClassifier(n_estimators = 500)
RFC.fit(X_train, y_train)
prediction_RFC = RFC.predict(X_test)
print(classification_report(y_test, prediction_RFC))
'''Respuesta:
                precision    recall  f1-score   support

           6       0.90      0.95      0.93       140
           7       0.67      0.57      0.62        28
           8       0.00      0.00      0.00         3

    accuracy                           0.87       171
   macro avg       0.52      0.51      0.51       171
weighted avg       0.85      0.87      0.86       171'''
print(confusion_matrix(y_test, prediction_RFC))
'''[[133   7   0] - 6
    [ 12  16   0] - 7
    [  2   1   0]] - 8'''
# Con este subset ponderado vemos un mejoramiento en el comportamiento del modelo, 
# sin embargo, el modelo sólo funcionaría bien con -quality- >= 6.
# Por lo tanto, tendremos que ajustar el modelo a partir del frame original, 
# donde el valor de -quality- lo haremos binario, que a partir de 6 o 7 sea 1 y el resto 0.
# De las matrices de confusión (en la anterior), tenemos que 134 clasificados en valor 6 
# estuvieron bien y 7 mal, asimismo, 12 en 7 bien y 16 mal, 2 en 8 bien y 1 mal. 
# Asi, el modelo esta sesgado hacia el promedio de los datos puesto que es muy dispar la cantidad de valores en 6 y el resto.

# En consiguiente, haremos un subset con respuesta binaria en calidad si es mayor o igual a 7
n = 0
for x in wine['quality']:
    if x >= 7:
        wine.at[n, 'quality'] = 1
    else:
        wine.at[n, 'quality'] = 0
    n = n+1
# Repetimos entreno, test y RFC.
X = wine.drop('quality', axis=1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 16)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
RFC = RandomForestClassifier(n_estimators = 50)
RFC.fit(X_train, y_train)
prediction_RFC = RFC.predict(X_test)
print(classification_report(y_test, prediction_RFC))
'''
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       149
           1       0.64      0.64      0.64        11

    accuracy                           0.95       160
   macro avg       0.80      0.80      0.80       160
weighted avg       0.95      0.95      0.95       160'''
print(confusion_matrix(y_test, prediction_RFC))
'''[[145   4]
    [  4   7]]'''
# Ajustando varias veces los valores, por ejemplo, cambiando la clasificación del -quality- de 6 a 7 
# (subiendo el umbral), un tamaño de testeo menor (10%) y 50 árboles de desición poco a poco fue mejorando el modelo, 
# con una calificación mayor, sin embargo, en la matriz vemos que funciona bastante bien para clasificar los 0, pero malo para los 1. 
# Habrá que intentar otro método. Antes, vemos que sin escalar, obtenemos cercanamente lo mismo.
X = wine.drop('quality', axis=1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 16)
X_train[:10]
RFC = RandomForestClassifier(n_estimators = 500)
RFC.fit(X_train, y_train)
prediction_RFC = RFC.predict(X_test)
print(classification_report(y_test, prediction_RFC))
'''precision    recall  f1-score   support

           0       0.98      0.95      0.97       149
           1       0.53      0.73      0.62        11

    accuracy                           0.94       160
   macro avg       0.76      0.84      0.79       160
weighted avg       0.95      0.94      0.94       160'''
print(confusion_matrix(y_test, prediction_RFC))
'''[[142   7]
    [  3   8]]'''
# ---------------------------------------------------------------------------------------------------#
# Dados los resultados de la matriz y, antes de pasar al siguiente modelo, 
# procederemos a estudiar los datos mediante un MANOVA, de esta forma 
# podremos observar cómo las variables influyen en la calidad y qué tanto.
# El propósito de usar un MANOVA es para ver el comportamiento de las varianzas
# de las variables y cómo influyen unas entre sí.
wine = pd.read_csv('winequality-red.csv')
wine.columns = wine.columns.str.replace(' ', '_') 
manova = MANOVA.from_formula('fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol ~ quality', data = wine)
result = manova.mv_test()
print(result)
# Los resultados serán colocados en un doc aparte del mismo repo.
# Dado que los resultados son distintos a los del MANOVA de R, sobre todo
# las interacciones, se añadirá al mismo repo esa parte en R.
# De los resultados del MANOVA en R tenemos que las variables que se interrelacionan
# en buena medida entre sí con la variable respuesta (Y) 'quality' son
# 'volatile acidity', 'residual sugar', chlorides', 'total sulfure dioxide', 'sulphates'
# y 'alcohol' (***). En menor medida 'free sulfur dioxide' y 'pH' (*).
# Con estos datos, quitaremos las variables que no mostraron significancia 
# en los siguientes algoritmos.
wine_man = pd.read_csv('winequality-red.csv')
n = 0
for x in wine_man['quality']:
    if x >= 7:
        wine_man.at[n, 'quality'] = 1
    else:
        wine_man.at[n, 'quality'] = 0
    n = n+1
wine_man = wine_man.drop(columns=['fixed acidity', 'citric acid', 
                                  'free sulfur dioxide', 'density'], axis=0)
# Repetimos el RFC con las columnas con mayor relevancia
X = wine_man.drop('quality', axis=1)
y = wine_man['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 16)
RFC = RandomForestClassifier(n_estimators = 600)
RFC.fit(X_train, y_train)
prediction_RFC = RFC.predict(X_test)
print(classification_report(y_test, prediction_RFC))
'''             precision    recall  f1-score   support

           0       0.99      0.96      0.97        75
           1       0.57      0.80      0.67         5

    accuracy                           0.95        80
   macro avg       0.78      0.88      0.82        80
weighted avg       0.96      0.95      0.95        80
'''
print(confusion_matrix(y_test, prediction_RFC))
'''[[[72  3]
     [ 1  4]]'''
# En general, es bueno prediciendo vino malo pero malo en predecir vino bueno.
# Acoplando los resultados del MANOVA, vemos que al eliminar las variables
# que no aportan en gran medida a la relación entre las variables con la respuesta
# hay una ligera mejoria en el desempeño del RFC. Por lo tanto, este será el set
# de datos que utilizaremos.
# ---------------------------------------------------------------------------------------------------#
# Segunda metodología: SVM Classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print(classification_report(y_test, pred_clf,  zero_division=1))
'''              precision    recall  f1-score   support

           0       0.93      1.00      0.96       149
           1       1.00      0.00      0.00        11

    accuracy                           0.93       160
   macro avg       0.97      0.50      0.48       160
weighted avg       0.94      0.93      0.90       160
'''
print(confusion_matrix(y_test, pred_clf))
'''[[149   0]
    [ 11   0]]'''
# Mejor desempeño que el RFC en predecir los buenos. SVM es más facil
# y rápido de utilizar con data raw, sus diferencias son en como tratamos
# la información antes.
# ---------------------------------------------------------------------------------------------------#
# Tercera metodología: Neural Network
# Puede tratarse con mucha data (huge), aguanta análisis de texto, puede procesar
# de muchas fuentes. Muchos tipos de procesamiento, pero necesita mucha data.
# hidden_layer_sizes cuantas capas tendrá la red, entre más capas mas recursos se necesitan.
# y cuantas veces se van a iterar en estas capas. Son tres capas de 11 clasificadores cada uno.
mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
print(classification_report(y_test, pred_mlpc,  zero_division=1))
'''              precision    recall  f1-score   support

           0       0.98      0.97      0.97       149
           1       0.62      0.73      0.67        11

    accuracy                           0.95       160
   macro avg       0.80      0.85      0.82       160
weighted avg       0.95      0.95      0.95       160
'''
print(confusion_matrix(y_test, pred_mlpc))
'''[[144   5]
    [  3   8]]'''
# Con estos resultados podemos concluir que es muy poca información
# para utilizar MLPC. 
# ---------------------------------------------------------------------------------------------------#
# Globales:
# Modelo    macro avg   weighted avg
# RFC:          0.78        0.96
# SVM:          0.97        0.94
# MLPC:         0.80        0.95
# ---------------------------------------------------------------------------------------------------#
predicts = [prediction_RFC, pred_clf, pred_mlpc]
for preds in predicts:
    cm = accuracy_score(y_test, preds)
    print(cm)
# con el accuracy_score 
# 0.95625
# 0.93125
# 0.925