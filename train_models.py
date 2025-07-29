"""
Instancia los modelos, serializando a disco los modelos ya entrenados.

1) Ejecución desde el terminal:

Los parámetros necesarios se recogen del archivo de configuración recibido como argumento a través del terminal:

--path_config: Path del archivo de configuración (ej.: "config\\train.conf")

2) Ejecución desde fuera del terminal:

Los parámetros necesarios se recogen del archivo 'train.conf' situado en el directorio 'config'
"""

import pandas as pd

from models.random_forest_clf import RandomForestClf
from models.random_forest_reg import RandomForestReg
from models.lda import Lda
from models.kmeans import Kmeans
from models.knn import Knn
from models.regresion_lineal import RegresionLineal
from models.mlp import Mlp

import argparse
import configparser
import sys

# archivo train.conf
SECCION_CONFIG = 'DEFAULT'
FICHERO_LOAD_DATOS = 'file_eda'
PATH_SAVE_DATOS = 'path_processed'
SECCION_FILE_TRAINED = 'FilesTrained'

# localización del archivo (para cuando se ejecuta este script desde fuera del terminal)
FILE_CONFIG = 'config/train.conf'


def load_data(file_path):
    """
    Lectura de disco del archivo
    :file_path: directorio y nombre del archivo
    :return: dataframe de los datos
    """
    print('Cargando datos...')

    try:
        df = pd.read_csv(file_path, index_col=0)
    except IOError as e:
        raise IOError('Error al intentar abrir el archivo ' + str(file_path) + ': ', e)

    return df


def aplicar_random_forest_clf(df, directory_save, variable):
    """
    Aplica el modelo Random Forest Classifier
    :df_rf: dataframe con los datos
    :directory_save_rf: directorio donde guardar los datos entrenados
    :variable_rf: variable objetivo sobre la que aplicar Random Forest: category, comments, ratings
    """
    # instancia el modelo
    model_rf = RandomForestClf()

    # entrena el modelo
    print()
    print(f'Entrenando el modelo Random Forest Classifier para la variable objetivo "{variable}"...')
    model_rf.fit(df, variable)

    # guarda los datos
    print('Guardando el modelo Random Forest...')
    model_rf.save(directory_save, variable)


def aplicar_random_forest_reg(df, directory_save, variable):
    """
    Aplica el modelo Random Forest Regressor
    :df: dataframe con los datos
    :directory_save: directorio donde guardar los datos entrenados
    :variable: variable objetivo sobre la que aplicar Random Forest: ratio_likes_dislikes
    """
    # instancia el modelo
    model_rf = RandomForestReg()

    # entrena el modelo
    print()
    print(f'Entrenando el modelo Random Forest Regressor para la variable objetivo "{variable}"...')
    model_rf.fit(df, variable)

    # guarda los datos
    print('Guardando el modelo Random Forest...')
    model_rf.save(directory_save, variable)


def aplicar_lda(df, directory_save):
    """
    Aplica el modelo LDA
    :df: dataframe con los datos
    :directory_save: directorio donde guardar los datos entrenados
    """
    # instancia el modelo
    model_lda = Lda()

    # entrena el modelo
    print()
    print('Entrenando el modelo LDA...')
    model_lda.fit(df)

    # guarda los datos
    print('Guardando el modelo LDA...')
    model_lda.save(directory_save)


def aplicar_kmeans(df, directory_files):
    """
    Aplica el modelo K-Means
    :df: dataframe con los datos
    :directory_files: directorio donde leer/guardar los datos entrenados
    """
    # instancia el modelo
    model_kmeans = Kmeans()

    # entrena el modelo
    print()
    print('Entrenando el modelo K-Means...')
    model_kmeans.fit(df, directory_files)

    # guarda los datos
    print('Guardando el modelo K-Means...')
    model_kmeans.save(directory_save)


def aplicar_knn(df, directory_save):
    """
    Aplica el modelo KNN (recomendador de vídeos)
    :df: dataframe con los datos
    :directory_save: directorio donde guardar los datos entrenados
    """
    # instancia el modelo
    model_knn = Knn()

    # entrena el modelo
    print()
    print('Entrenando el modelo KNN...')
    model_knn.fit(df)

    # guarda los datos
    print('Guardando el modelo KNN...')
    model_knn.save(directory_save)


def aplicar_regresion_lineal(df, directory_save):
    """
    Aplica el modelo Linear Regression
    :df: dataframe con los datos
    :directory_save: directorio donde guardar los datos entrenados
    """
    # instancia el modelo
    model_rl = RegresionLineal()

    # entrena el modelo
    print()
    print(f'Entrenando el modelo Regresión Lineal...')
    model_rl.fit(df)

    # guarda los datos
    print('Guardando el modelo Regresión Lineal...')
    model_rl.save(directory_save)


def aplicar_mlp(df, directory_save):
    """
    Aplica el modelo MLP
    :df: dataframe con los datos
    :directory_save: directorio donde guardar los datos entrenados
    """
    # instancia el modelo
    model_mlp = Mlp()

    # entrena el modelo
    print()
    print(f'Entrenando el modelo MLP para la variable objetivo "comments_disabled"...')
    model_mlp.fit(df)

    # guarda los datos
    print('Guardando el modelo MLP...')
    model_mlp.save(directory_save)

# ******************************* inicio script principal **********************************************

if __name__ == '__main__':

    # carga el archivo de configuración
    if sys.stdin.isatty():
        # el script sí se está ejecutando desde el terminal

        # llamada desde el terminal
        # python train_models.py --path_config config\train.conf

        parser = argparse.ArgumentParser(description="Instanciar y entrenar modelos")

        # argumento que el usuario pasa al script
        parser.add_argument('--path_config', required=True,
                            help=r'Path del archivo de configuración (ej.: "config\train.conf")')
        args = parser.parse_args()

        # accede al archivo de configuración
        config = configparser.ConfigParser()
        config.read(args.path_config)

    else:
        # el script se está ejecutando desde fuera del terminal

        # accede al archivo de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

    # directorio donde se guardarán los datos preprocesados
    directory_save = config[SECCION_CONFIG][PATH_SAVE_DATOS]

    # carga los datos
    df = load_data(config[SECCION_CONFIG][FICHERO_LOAD_DATOS])

    # aplica los modelos

    # notebook 3: clasificación categoría, comments_disabled, ratings_disabled (Random Forest)

    # Random Forest - category
    aplicar_random_forest_clf(df, directory_save, 'category')

    # Random Forest - comments_disabled
    aplicar_random_forest_clf(df, directory_save, 'comments')

    # Random Forest - ratings_disabled
    aplicar_random_forest_clf(df, directory_save, 'ratings')

    # MLP - comments_disabled
    # no ofrece mejores resultados que Random Forest, pero lo industrializo por tener también una versión de MLP
    aplicar_mlp(df, directory_save)

    # notebook 4: regresión likes (Regresión Lineal), ratio likes/dislikes (Random Forest)

    # Regresión Lineal
    aplicar_regresion_lineal(df, directory_save)

    # Random Forest - ratio likes/dislikes
    aplicar_random_forest_reg(df, directory_save, 'ratio_likes_dislikes')

    # notebook 5: clusterización (K-Means sobre LDA)

    # LDA
    aplicar_lda(df, directory_save)

    # K-Means
    aplicar_kmeans(df, directory_save)

    # notebook 6: recomendador (KNN)
    aplicar_knn(df, directory_save)

    print()
    print('Acabado')
