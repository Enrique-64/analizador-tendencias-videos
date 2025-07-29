"""
Instancia el modelo solicitado por el usuario a través del terminal.
Carga el modelo ya entrenado y realiza inferencia de nuevos datos.

Argumentos de entrada (desde el terminal):

1) --model: Nombre del modelo a utilizar (ej.: "RandomForest")
2) --file: Ruta "en crudo" del archivo con los datos originales (ej.: "C:\\Python\\datos\\csv_pruebas.csv")
3) --weights: Ruta "en crudo" del archivo pickle con los pesos (ej.: "C:\\Python\\datos\\weights.pkl").
Parámetro opcional.
"""

import pickle
import pandas as pd

import argparse
import configparser

from models.random_forest_clf import RandomForestClf
from models.random_forest_reg import RandomForestReg
from models.lda import Lda
from models.kmeans import Kmeans
from models.knn import Knn
from models.regresion_lineal import RegresionLineal
from models.mlp import Mlp

# archivo train.conf
FILE_CONFIG = 'config/train.conf'
SECCION_CONFIG = 'DEFAULT'
PATH_LOAD_TRAIN = 'path_processed'

# archivos entrenados de los modelos
FILE_TRAINED_RANDOM_FOREST = 'random_forest'
FILE_TRAINED_LDA = 'lda'
FILE_TRAINED_KMEANS = 'kmeans'
FILE_TRAINED_KNN = 'knn'
FILE_TRAINED_REGRESION_LINEAL = 'regresion_lineal'
FILE_TRAINED_MLP = 'mlp'

# lista de los modelos implementados
AVAILABLE_MODELS = ['RandomForest', 'LDA', 'K-Means', 'KNN', 'RegresiónLineal', 'MLP']


def instanciar_random_forest_clf(df, weights, directory_load, variable):
    """
    Instancia Random Forest Classifier con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos (sin uso en Random Forest)
    :directory_load: directorio de los datos ya entrenados
    :variable: variable objetivo
    """

    # instancia el modelo
    print()
    print('Random Forest Classifier')
    model_rf = RandomForestClf()

    # carga el método entrenado desde el archivo
    print(f'Cargando datos de entrenamiento para la variable "{variable}"...')
    model_rf.load(directory_load, variable)

    # realiza la predicción de nuevos datos
    print(f'Predicción de la variable "{variable}" a partir de los nuevos datos...')
    predictions = model_rf.predict(df, variable)

    print('Predicciones:', predictions)
    print('Número de predicciones:', predictions.shape[0])


def instanciar_random_forest_reg(df, weights, directory_load, variable):
    """
    Instancia Random Forest Regressor con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos (sin uso en Random Forest)
    :directory_load: directorio de los datos ya entrenados
    :variable: variable objetivo
    """

    # instancia el modelo
    print()
    print('Random Forest Regressor')
    model_rf = RandomForestReg()

    # carga el método entrenado desde el archivo
    print(f'Cargando datos de entrenamiento para la variable "{variable}"...')
    model_rf.load(directory_load, variable)

    # realiza la predicción de nuevos datos
    print(f'Predicción de la variable "{variable}" a partir de los nuevos datos...')
    predictions = model_rf.predict(df, variable)

    print('Predicciones:', predictions)
    print('Número de predicciones:', predictions.shape[0])


def instanciar_lda(df, weights, directory_load):
    """
    Instancia LDA con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos (sin uso en LDA)
    :directory_load: directorio de los datos ya entrenados
    """

    # instancia el modelo
    print()
    print('LDA')
    model_lda = Lda()

    # carga el método entrenado desde el archivo
    print('Cargando datos de entrenamiento...')
    model_lda.load(directory_load)

    # realiza la predicción de nuevos datos
    print('Predicción a partir de los nuevos datos...')
    predictions = model_lda.predict(df)

    print('Predicciones:', predictions)
    print('Número de predicciones:', predictions.shape[0])


def instanciar_kmeans(df, weights, directory_load):
    """
    Instancia K-Means con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos (sin uso en K-Means)
    :directory_load: directorio de los datos ya entrenados
    """

    # instancia el modelo
    print()
    print('K-Means')
    model_kmeans = Kmeans()

    # carga el método entrenado desde el archivo
    print('Cargando datos de entrenamiento...')
    model_kmeans.load(directory_load)

    # realiza la predicción de nuevos datos
    print('Predicción a partir de los nuevos datos...')
    predictions = model_kmeans.predict(df)

    print()
    print('Predicciones:', predictions)
    print('Número de predicciones:', predictions.shape[0])


def instanciar_knn(df, weights, directory_load):
    """
    Instancia KNN con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos (sin uso en KNN)
    :directory_load: directorio de los datos ya entrenados
    """

    # instancia el modelo
    print()
    print('KNN')
    model_knn = Knn()

    # carga el método entrenado desde el archivo
    print('Cargando datos de entrenamiento...')
    model_knn.load(directory_load)

    # pregunta al usuario qué quiere buscar
    print()
    titulo = input('Introduzca un título o un texto a buscar: ')
    n_neighbors = int(input('Introduzca el número de vídeos recomendados: '))

    # realiza la recomendación de vídeos
    recomendaciones = model_knn.recomendador(titulo, df, n_neighbors)

    print('Recomendaciones:')
    for i in recomendaciones:
        print('Vídeo', i[0], '(dist = {:.2}): '.format(i[1]), df.iloc[i[0]]['title'])


def instanciar_regresion_lineal(df, weights, directory_load):
    """
    Instancia Regresión Lineal con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos (sin uso en Regresión Lineal)
    :directory_load: directorio de los datos ya entrenados
    """

    # instancia el modelo
    print()
    print('Regresión Lineal')
    model_rl = RegresionLineal()

    # carga el método entrenado desde el archivo
    print(f'Cargando datos de entrenamiento...')
    model_rl.load(directory_load)

    # realiza la predicción de nuevos datos
    print(f'Predicción a partir de los nuevos datos...')
    predictions = model_rl.predict(df)

    print('Predicciones:', predictions)
    print('Número de predicciones:', predictions.shape[0])


def instanciar_mlp(df, weights, directory_load):
    """
    Instancia MLP con los datos proporcionados
    :df: dataframe con los datos a instanciar
    :weights: objeto pickle de pesos para el modelo; None si no hay pesos
    :directory_load: directorio de los datos ya entrenados
    """

    # instancia el modelo
    print()
    print('MLP')
    model_mlp = Mlp()

    # carga el método entrenado desde el archivo
    print('Cargando datos de entrenamiento...')
    model_mlp.load(directory_load)

    # realiza la predicción de nuevos datos
    print('Predicción a partir de los nuevos datos...')
    predictions = model_mlp.predict(df, weights)

    print('Predicciones:', predictions)
    print('Número de predicciones:', predictions.shape[0])


# *********************************** inicio script principal ************************************

# llamada desde el terminal
# python inference_model.py --model RandomForest --weights C:\Python\datos\file1.pkl --file C:\Python\datos\file1.csv
parser = argparse.ArgumentParser(description="Instanciar modelos con nuevos datos")

# argumentos que el usuario pasa al script
parser.add_argument('--model', required=True,
                    help=r'Nombre del modelo a utilizar (ej.: "RandomForest")')
parser.add_argument('--file', required=True,
                    help=r'Ruta "en crudo" del archivo con los datos originales (ej.: "C:\Python\datos\csv_pruebas.csv")')
parser.add_argument('--weights', required=False,
                    help=r'Ruta "en crudo" del archivo pickle con los pesos (ej.: "C:\Python\datos\weights.pkl")')

args = parser.parse_args()

# comprueba el modelo recibido
if args.model in AVAILABLE_MODELS:
    # modelo válido

    # comprueba los archivos recibidos
    try:
        df = pd.read_csv(args.file, index_col=0)
    except IOError as e:
        raise IOError('Error al intentar abrir el archivo ' + str(args.file) + ': ', e)

    if args.weights is not None:
        # recibido fichero de pesos desde el terminal
        try:
            # abre el fichero
            with (open(args.weights, 'rb') as weights_file):
                try:
                    # carga el objeto pickle
                    weights = pickle.load(weights_file)
                except Exception as e:
                    raise Exception('Error al cargar de disco el objeto pickle:"', e, '"en fichero"', weights_file, '"')
        except Exception as e:
            raise Exception('Error al intentar abrir el archivo ' + str(args.weights) + ': ', e)
    else:
        # no recibido fichero de pesos desde el terminal
        weights = None

    # obtiene el directorio con los archivos de entrenamiento

    # accede al archivo de configuración
    config = configparser.ConfigParser()
    config.read(FILE_CONFIG)

    # directorio donde se guardan los archivos de entrenamiento
    directory_load = config[SECCION_CONFIG][PATH_LOAD_TRAIN]

    # instancia el modelo solicitado
    match args.model:
        case 'RandomForest':

            # category
            instanciar_random_forest_clf(df, weights, directory_load, 'category')

            # comments_disabled
            instanciar_random_forest_clf(df, weights, directory_load, 'comments')

            # ratings_disabled
            instanciar_random_forest_clf(df, weights, directory_load, 'ratings')

            # ratio_likes_dislikes
            instanciar_random_forest_reg(df, weights, directory_load, 'ratio_likes_dislikes')

        case 'LDA':
            instanciar_lda(df, weights, directory_load)

        case 'K-Means':
            instanciar_kmeans(df, weights, directory_load)

        case 'KNN':
            instanciar_knn(df, weights, directory_load)

        case 'RegresiónLineal':
            instanciar_regresion_lineal(df, weights, directory_load)

        case 'MLP':
            instanciar_mlp(df, weights, directory_load)

else:
    # el modelo recibido como argumento de entrada no es válido
    print(f'El modelo {args.model} no es un modelo implementado.')
    print('Los modelos implementados son:')
    print(AVAILABLE_MODELS)
