"""
Modelo K-Means de Sklearn

Clase: Kmeans
"""

import pickle
import configparser

from sklearn.cluster import KMeans

from models.lda import Lda
from utils.data_utils import evaluar_modelo_y_pred

# fichero train.conf
FILE_CONFIG = 'config/train.conf'
SECCION_CONFIG = 'KMeans'
SECCION_FILE_TRAINED = 'FilesTrained'
FILE_TRAINED = 'kmeans'

class Kmeans:
    """
    Métodos:
    fit: Entrenamiento del modelo
    predict: Predicciones a nuevos datos a partir del modelo ya entrenado
    save: Guarda en disco serializado en pickle el objeto Kmeans actual
    load: Carga de disco un objeto Kmeans con el entrenamiento del modelo
    __preprocess_data: Preprocesamiento del dataset antes de aplicar el modelo
    __obtener_nombre_archivo: Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
    """

    def __init__(self):

        # inicializa el modelo de K-Means
        self.model = None

        # inicializa el modelo de LDA
        self.lda = None

    def fit(self, df, directory_load):
        '''
        Entrenamiento del modelo
        :df: dataframe de Pandas con el dataset
        :directory_load: directorio donde leer los datos entrenados
        '''

        # fichero de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

        # preprocesamiento de datos
        X_train, y_train = self.__preprocess_data(df)

        # carga modelo LDA entrenado

        # instancia el modelo
        print('Datos de entrenamiento de LDA...')
        model_lda = Lda()

        # carga el método entrenado desde el archivo
        model_lda.load(directory_load)

        # obtenemos los resultados de LDA
        self.lda = model_lda.transform_lda(X_train)

        print('Aplicando K-Means...')

        # parámetros del modelo
        # n_clusters: int
        n_clusters = config.getint(SECCION_CONFIG, 'n_clusters')
        # random_state: int
        random_state = config.getint(SECCION_CONFIG, 'random_state')

        params_km = {'n_clusters': n_clusters, 'random_state': random_state}

        # instancia K-Means
        self.model = KMeans(**params_km)

        # entrena el modelo sobre los datos proyectados en el espacio reducido de LDA
        self.model.fit(self.lda)

        # evalúa el modelo
        predictions = self.model.predict(self.lda)
        evaluar_modelo_y_pred(self.model, y_train, predictions)

    def predict(self, df):
        '''
        Predicciones a nuevos datos a partir del modelo ya entrenado.
        :df: DataFrame con los nuevos datos
        :return: predicciones
        '''

        # verifica que el modelo esté cargado
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado. Use el método 'load' para cargarlo.")

        # preprocesamiento de datos
        X_train, y_train = self.__preprocess_data(df)

        # realiza la predicción utilizando el modelo cargado
        predictions = self.model.predict(self.lda)

        # evalúa la predicción
        evaluar_modelo_y_pred(self.model, y_train, predictions)

        # devuelve la predicción
        return predictions

    def save(self, directory_save):
        '''
        Guarda en disco serializado en pickle el objeto Kmeans actual
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
        Carga de disco un objeto Kmeans con el entrenamiento del modelo
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
        Preprocesamiento del dataset antes de aplicar el modelo.
        :df: DataFrame de Pandas
        :return: X (variables de entrada), y (variable de salida)
        '''

        # selecciona las variables de entrada de entre las columnas del dataset
        X = df[['channel_title_encoded', 'tags_encoded', 'views_encoded', 'likes_encoded', 'dislikes_encoded',
                'comment_count_encoded', 'comments_disabled', 'ratings_disabled', 'country_encoded',
                'trending_date_year_encoded', 'trending_date_month_encoded', 'publish_time_year_encoded',
                'publish_time_month_encoded', 'ratio_viralidad', 'ratio_interaccion_dia_encoded',
                'publish_trending_days']]

        # selecciona la variable de salida
        y = df[['category_encoded']].copy(deep=True)

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
