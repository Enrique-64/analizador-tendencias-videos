"""
Funciones auxiliares que pueden ser reutilizadas en varios scripts
"""
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def muestrear_datos(X, y, sample_size):
    '''
    Muestrea al porcentaje indicado los dos dataframes recibidos
    :X: dataframe con un conjunto de variables de entradas
    :y: dataframe con un conjunto de la variable de salida
    :sample_size: porcentaje de muestreo
    :return: X_sampled, y_sampled, dataframes muestreados
    '''
    X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=sample_size, random_state=42)
    return X_sampled, y_sampled

def evaluar_modelo_train_test(clf, X_train, y_train, X_test, y_test):
    '''
    Evalúa un modelo de clasificación
    :clf: modelo a evaluar
    :X_train, y_train, X_test, y_test: conjuntos de entrenamiento y pruebas
    '''
    # conjunto de entrenamiento
    pred = clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred)
    print(f"Accuracy Score Train: {accuracy_train * 100:.2f}%")

    # conjunto de pruebas
    pred = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, pred)
    print(f"Accuracy Score Test: {accuracy_test * 100:.2f}%")

def evaluar_modelo_y_pred(clf, y, predictions):
    '''
    Evalúa un modelo de clasificación a partir de la variable objetivo y la predicción
    :clf: modelo a evaluar
    :y: variable objetivo
    :predictions: predicción del modelo
    '''
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy Score Train: {accuracy * 100:.2f}%")

def evaluar_modelo_regresion(reg, X_train, y_train, X_test, y_test):
    '''
    Evalúa un modelo de regresión a partir del conjunto de entrenamiento
    :reg: modelo a evaluar
    :X_train, y_train, X_test, y_test: conjuntos de entrenamiento y pruebas
    '''

    # Train
    score_train = reg.score(X_train, y_train)
    print('Score Train: {:.2%}'.format(score_train))

    # Test
    score_test = reg.score(X_test, y_test)
    print('Score Test: {:.2%}'.format(score_test))

def evaluar_modelo_regresion_y_pred(reg, y, predictions):
    '''
    Evalúa un modelo de regresión a partir de la variable objetivo y la predicción
    :reg: modelo a evaluar
    :y: variable objetivo
    :predictions: predicción del modelo
    '''
    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

    # mae, mse, rmse

    mae = mean_absolute_error(y, predictions)
    print('Mean Absolute Error: {:.2%}'.format(mae))

    mse = mean_squared_error(y, predictions)
    print('Mean Squared Error: {:.2%}'.format(mse))

    rmse = root_mean_squared_error(y, predictions)
    print('Root Mean Squared Error: {:.2%}'.format(rmse))

def balancear_pesos(y_train, variable):
    '''
    Balancea la variable objetivo normalizando la suma de los pesos de la muestra para cada clase al mismo valor
    :y_train: conjunto de entrenamiento de la variable objetivo
    :variable: nombre de la variable objetivo
    :return: diccionario con los pesos de cada clase
    '''
    # número de muestras por clase
    class_distribution = Counter(y_train[variable])
    # número total de muestras
    total_samples = sum(class_distribution.values())
    # número de clases
    num_classes = len(class_distribution)
    # suma de pesos por clase
    sum_per_class = total_samples / num_classes
    # pesos de cada clase
    class_weight = {}
    for class_name, count in class_distribution.items():
        class_weight[class_name] = sum_per_class / count
    return class_weight

def balancear_pesos_percentiles(X_train, y_train, variable, perc_sup, perc_inf):
    '''
    Balancea los conjuntos de entrenamiento aplicando oversampling en las clases minoritarias (definidas por un
    percentil inferior) y undersampling en las mayoritarias (definidas por un percentil superior)
    :X_train: conjunto de entrenamiento de las variables de entrada
    :y_train: conjunto de entrenamiento de la variable objetivo
    :variable: nombre de la variable objetivo
    :perc_sup: percentil superior
    :perc_inf: percentil inferior
    :return: X_train, y_train balanceados
    '''

    # número de muestras por clase
    class_distribution = Counter(y_train[variable])
    class_counts = np.array(list(class_distribution.values()))

    # percentiles superior e inferior que van a definir las clases a modificar
    percentile_sup = np.percentile(class_counts, perc_sup)
    percentile_inf = np.percentile(class_counts, perc_inf)

    # estrategias de oversampling y undersampling
    oversample_strategy = {cls: int(percentile_inf) for cls, count in class_distribution.items() if
                           count < percentile_inf}
    undersample_strategy = {cls: int(percentile_sup) for cls, count in class_distribution.items() if
                           count > percentile_sup}

    # SMOTE utiliza el parámetro n_neighbors por defecto a 5, vamos a comprobar si el número mínimo de muestras de las
    # clases es inferior o igual a este valor, en cuyo caso hay que reducirlo
    # valor mínimo de muestras de las clases
    min_samples_class = y_train.value_counts().min()
    # se selecciona el n_neighbors necesario
    n_neighbors = min(min_samples_class - 1, 5)

    # SMOTE para oversampling en las clases minoritarias
    smote = SMOTE(sampling_strategy=oversample_strategy, random_state=42, k_neighbors=n_neighbors)
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    # RandomUnderSampler para las clases mayoritarias
    undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
    X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_over, y_train_over)

    return X_train_balanced, y_train_balanced
