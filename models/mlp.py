"""
Modelo MLP de Sklearn

Clase: Mlp
"""

import numpy as np
import pickle
import configparser
import ast

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils.data_utils import evaluar_modelo_train_test, evaluar_modelo_y_pred, balancear_pesos_percentiles

# fichero train.conf
FILE_CONFIG = 'config/train.conf'
SECCION_CONFIG = 'Mlp'
SECCION_FILE_TRAINED = 'FilesTrained'
FILE_TRAINED = 'mlp'

class Mlp:
    """
    Métodos:
    fit: Entrenamiento del modelo
    predict: Predicciones a nuevos datos a partir del modelo ya entrenado
    save: Guarda en disco serializado en pickle el objeto Mlp actual
    load: Carga de disco un objeto Mlp con el entrenamiento del modelo
    __preprocess_data: Preprocesamiento del dataset antes de aplicar el modelo
    __obtener_nombre_archivo: Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
    """

    def __init__(self):

        # inicializa el modelo
        self.model = None
        # inicializamos los pesos
        self.weights = None

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

        # parámetros del modelo

        # activation: str
        activation = config.get(SECCION_CONFIG, 'activation')
        # alpha: float
        alpha = config.getfloat(SECCION_CONFIG, 'alpha')
        # batch_size: str
        batch_size = config.get(SECCION_CONFIG, 'batch_size')
        # early_stopping: bool
        early_stopping = config.getboolean(SECCION_CONFIG, 'early_stopping')
        # hidden_layer_sizes: array
        hidden_layer_sizes = config.get(SECCION_CONFIG, 'hidden_layer_sizes')
        try:
            # intenta convertirlo a tupla
            hidden_layer_sizes = ast.literal_eval(hidden_layer_sizes)
        except ValueError:
            raise Exception('Error en el parámetro "hidden_layer_sizes", debe ser de tipo "tuple"')
        # learning_rate: str
        learning_rate = config.get(SECCION_CONFIG, 'learning_rate')
        # max_iter: int
        max_iter = config.getint(SECCION_CONFIG, 'max_iter')
        # n_iter_no_change: int
        n_iter_no_change = config.getint(SECCION_CONFIG, 'n_iter_no_change')
        # shuffle: bool
        shuffle = config.getboolean(SECCION_CONFIG, 'shuffle')
        # random_state: int
        random_state = config.getint(SECCION_CONFIG, 'random_state')

        params_mlp = {'activation': activation, 'alpha': alpha, 'batch_size': batch_size,
                      'early_stopping': early_stopping, 'hidden_layer_sizes': hidden_layer_sizes,
                      'learning_rate': learning_rate, 'max_iter': max_iter, 'n_iter_no_change': n_iter_no_change,
                      'shuffle': shuffle, 'random_state': random_state}

        # perc_sup: float
        perc_sup = config.getfloat(SECCION_CONFIG, 'perc_sup')
        # perc_inf: float
        perc_inf = config.getfloat(SECCION_CONFIG, 'perc_inf')

        # balancea los conjuntos de entrenamiento

        X_train_balanced, y_train_balanced = balancear_pesos_percentiles(X_train, y_train, 'comments_disabled',
                                                                         perc_sup, perc_inf)

        # instancia MLP
        self.model = MLPClassifier(**params_mlp)

        # entrena el modelo
        self.model.fit(X_train_balanced, np.ravel(y_train_balanced))

        # guarda pesos y términos independientes
        self.weights = {
            'coefs_': self.model.coefs_,
            'intercepts_': self.model.intercepts_
        }

        # evalúa el modelo
        evaluar_modelo_train_test(self.model, X_train, y_train, X_test, y_test)

    def predict(self, df, weights):
        '''
        Predicciones a nuevos datos a partir del modelo ya entrenado.
        :df: DataFrame con los nuevos datos
        :weights: objeto pickle de pesos para el modelo; None si no hay pesos
        :return: predicciones
        '''

        # verifica que el modelo esté cargado
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado. Use el método 'load' para cargarlo.")

        # preprocesa los datos recibidos
        X, y = self.__preprocess_data(df)

        if weights is not None:
            # recibidos nuevos pesos para la predicción
            self.weights = weights
            self.model.coefs_ = self.weights['coefs_']
            self.model.intercepts_ = self.weights['intercepts_']

        # realiza la predicción utilizando el modelo cargado
        predictions = self.model.predict(X)

        # evalúa la predicción
        evaluar_modelo_y_pred(self.model, y, predictions)

        # devuelve la predicción
        return predictions

    def save(self, directory_save):
        '''
        Guarda en disco serializado en pickle el objeto Mlp actual
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

        # guarda los pesos (para pruebas)
        path_file = directory_save + 'weights.pkl'
        with open(path_file, 'wb') as fp:
            pickle.dump(self.weights, fp)
        print('Pesos guardados en el fichero:', path_file)

    def load(self, directory_load):
        '''
        Carga de disco un objeto Mlp con el entrenamiento del modelo
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
        X = df[['channel_title_encoded', 'tags_encoded', 'views_encoded', 'likes_encoded', 'dislikes_encoded',
                'comment_count_encoded', 'ratings_disabled', 'category_encoded', 'country_encoded', 'ratio_viralidad',
                'ratio_interaccion_dia_encoded', 'trending_date_year_encoded', 'trending_date_month_encoded',
                'publish_time_year_encoded', 'publish_time_month_encoded', 'publish_trending_days']]

        # selecciona la variable de salida
        y = df[['comments_disabled']].copy(deep=True)

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
