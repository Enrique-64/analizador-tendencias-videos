"""
Modelo Regresión Lineal de Sklearn

Clase: RegresionLineal
"""

import numpy as np
import pickle
import configparser

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from utils.data_utils import evaluar_modelo_regresion, evaluar_modelo_regresion_y_pred

# fichero train.conf
FILE_CONFIG = 'config/train.conf'
SECCION_CONFIG = 'RegresionLineal'
SECCION_FILE_TRAINED = 'FilesTrained'
FILE_TRAINED = 'regresion_lineal'

class RegresionLineal:
    """
    Métodos:
    fit: Entrenamiento del modelo
    predict: Predicciones a nuevos datos a partir del modelo ya entrenado
    save: Guarda en disco serializado en pickle el objeto RegresionLineal actual
    load: Carga de disco un objeto RegresionLineal con el entrenamiento del modelo
    __preprocess_data: Preprocesamiento del dataset antes de aplicar el modelo
    __obtener_nombre_archivo: Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
    """

    def __init__(self):

        # inicializa el modelo
        self.model = None

    def fit(self, df):
        '''
        Entrenamiento del modelo
        :df: dataframe de Pandas con el dataset
        '''

        # fichero de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

        # preprocesamiento de datos
        X, y = self.__preprocess_data(df)

        # conjuntos de entrenamiento y pruebas
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # instancia Regresión Lineal
        self.model = LinearRegression()

        # entrena el modelo
        self.model.fit(X_train, np.ravel(y_train))

        # evalúa el modelo
        evaluar_modelo_regresion(self.model, X_train, y_train, X_test, y_test)

        print('- Train:')
        y_train_pred = self.model.predict(X_train)
        evaluar_modelo_regresion_y_pred(self.model, y_train, y_train_pred)

        print('- Test:')
        y_test_pred = self.model.predict(X_test)
        evaluar_modelo_regresion_y_pred(self.model, y_test, y_test_pred)

    def predict(self, df):
        '''
        Predicciones a nuevos datos a partir del modelo ya entrenado.
        :df: DataFrame con los nuevos datos
        :return: predicciones
        '''

        # verifica que el modelo esté cargado
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado. Use el método 'load' para cargarlo.")

        # preprocesa los datos recibidos
        X, y = self.__preprocess_data(df)

        # realiza la predicción utilizando el modelo cargado
        predictions = self.model.predict(X)

        # evalúa la predicción
        evaluar_modelo_regresion_y_pred(self.model, y, predictions)

        # devuelve la predicción
        return predictions

    def save(self, directory_save):
        '''
        Guarda en disco serializado en pickle el objeto RegresionLineal actual
        :directory_save: directorio de los datos ya entrenados
        '''

        # obtiene el nombre del archivo a guardar
        path_file = directory_save + self.__obtener_nombre_archivo()

        # abre archivo para escritura
        try:
            with open(path_file, 'wb') as fp:
                pickle.dump(self, fp)
            print('Datos guardados en el fichero:', path_file)
        except IOError as e:
            raise IOError('Error al intentar guardar los datos en el archivo' + str(path_file) + ':', e)

    def load(self, directory_load):
        '''
        Carga de disco un objeto RegresionLineal con el entrenamiento del modelo
        :directory_load: directorio de los datos ya entrenados
        '''

        # obtiene el nombre del archivo a leer
        path_file = directory_load + self.__obtener_nombre_archivo()

        # abre archivo para lectura
        try:
            with (open(path_file, 'rb') as model_file):
                try:
                    objeto = pickle.load(model_file)
                except Exception as e:
                    raise Exception('Error al cargar de disco el objeto pickle:"', e, '"en fichero"', path_file, '"')
                print('Datos de entrenamiento cargados de disco:', path_file)
        except IOError as e:
            raise IOError('Error al intentar abrir el archivo ' + str(path_file) + ': ', e)

        # actualiza los atributos de la instancia actual con el modelo cargado
        self.__dict__.update(objeto.__dict__)

    def __preprocess_data(self, df):
        '''
        Preprocesamiento del dataset antes de aplicar el modelo
        :df: DataFrame de Pandas
        :return: X (variables de entrada), y (variable de salida)
        '''

        # selecciona las variables de entrada de entre las columnas del dataset
        X = df[['channel_title_encoded', 'tags_encoded', 'views_encoded', 'dislikes_encoded', 'comment_count_encoded',
                'comments_disabled', 'ratings_disabled', 'category_encoded', 'country_encoded', 'ratio_viralidad',
                'ratio_interaccion_dia_encoded', 'publish_trending_days', 'trending_date_year_encoded',
                'trending_date_month_encoded', 'publish_time_year_encoded', 'publish_time_month_encoded']]

        # selecciona la variable de salida
        y = df[['likes_encoded']].copy(deep=True)

        # devuelve ambas variables
        return X, y

    def __obtener_nombre_archivo(self):
        '''
        Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
        :return: nombre del archivo (sin directorios)
        '''

        # archivo de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

        # lee y devuelve el nombre del archivo
        return config[SECCION_FILE_TRAINED][FILE_TRAINED]
