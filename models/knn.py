"""
Modelo KNN (NearestNeighbors) de Sklearn

Clase: Knn
"""

import pickle
import configparser

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

import string
from nltk.corpus import stopwords

# fichero train.conf
FILE_CONFIG = 'config/train.conf'
SECCION_CONFIG = 'KNN'
SECCION_FILE_TRAINED = 'FilesTrained'
FILE_TRAINED = 'KNN'

class Knn:
    """
    Métodos:
    fit: Entrenamiento del modelo
    recomendador: Recomendaciones de vídeos a partir del modelo ya entrenado
    save: Guarda en disco serializado en pickle el objeto Knn actual
    load: Carga de disco un objeto Knn con el entrenamiento del modelo
    __preprocess_data: Preprocesamiento del dataset antes de aplicar el modelo
    __obtener_nombre_archivo: Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
    """

    def __init__(self):

        # inicializa el modelo
        self.model = None
        # instancia del conversor del título
        self.vectorizer = None

    def fit(self, df):
        '''
        Entrenamiento del modelo
        :df: dataframe de Pandas con el dataset
        '''

        # fichero de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

        # preprocesamiento de datos
        df_recom, X = self.__preprocess_data(df)

        # parámetros del modelo
        # n_neighbors: int
        n_neighbors = config.getint(SECCION_CONFIG, 'n_neighbors')
        # metric: str
        metric = config.get(SECCION_CONFIG, 'metric')

        params_knn = {'n_neighbors': n_neighbors, 'metric': metric}

        # instancia NearestNeighbors
        self.model = NearestNeighbors(**params_knn)

        # entrena el modelo
        self.model.fit(X)

    def recomendador(self, titulo, df, n_neighbors=5):
        '''
        Recomendaciones de vídeos a partir del modelo ya entrenado
        :titulo: texto por el que realizar la búsqueda
        :df: DataFrame con los datos
        :n_neighbors: número vecinos a buscar
        :return: tupla (índice, distancia) con índices del dataframe y distancias a los vecinos
        '''

        # verifica que el modelo esté cargado
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado. Use el método 'load' para cargarlo.")

        # aplica el conversor al título a buscar
        titulo_vectorizado = self.vectorizer.transform([titulo])

        # añade los valores promedio al resto de características
        dummy_views = sp.csr_matrix([[df['views_encoded'].mean()]])
        dummy_likes = sp.csr_matrix([[df['likes_encoded'].mean()]])
        dummy_dislikes = sp.csr_matrix([[df['dislikes_encoded'].mean()]])
        dummy_comment_count = sp.csr_matrix([[df['comment_count_encoded'].mean()]])
        dummy_category = sp.csr_matrix([[df['category_encoded'].mean()]])
        dummy_ratio_viralidad = sp.csr_matrix([[df['ratio_viralidad'].mean()]])
        dummy_ratio_interaccion_dia = sp.csr_matrix([[df['ratio_interaccion_dia_encoded'].mean()]])

        # crea el vector completo a buscar
        vector_completo = sp.hstack((titulo_vectorizado, dummy_views, dummy_likes, dummy_dislikes, dummy_comment_count,
                                     dummy_category, dummy_ratio_viralidad, dummy_ratio_interaccion_dia))

        # busca los vecinos más cercanos
        distancias, indices = self.model.kneighbors(vector_completo, n_neighbors=n_neighbors)

        # lista de tuplas (índice, distancia) de los vecinos más cercanos
        neighbors = []
        for x1, x2 in zip(indices[0], distancias[0]):
            neighbors.append((x1, x2))

        # ordena por distancia de mayor a menor
        neighbors.sort(key=lambda x: x[1], reverse=True)

        return neighbors

    def save(self, directory_save):
        '''
        Guarda en disco serializado en pickle el objeto Knn actual
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
        Carga de disco un objeto Knn con el entrenamiento del modelo
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
        :variable: variable objetivo
        :return: df_recom (dataframe modificado), X (matriz de características)
        '''

        df_recom = df[['title', 'views_encoded', 'likes_encoded', 'dislikes_encoded', 'comment_count_encoded',
                       'category_encoded', 'ratio_viralidad', 'ratio_interaccion_dia_encoded']].copy()

        # quita palabras comunes en varios idiomas
        stop_words_en = stopwords.words('english')
        stop_words_fr = stopwords.words('french')
        stop_words_ge = stopwords.words('german')
        stop_words_be = stopwords.words('bengali')
        stop_words_sp = stopwords.words('spanish')
        stop_words_ru = stopwords.words('russian')

        stop_words_multi = list(
            set(stop_words_en + stop_words_fr + stop_words_ge + stop_words_be + stop_words_sp + stop_words_ru))

        # quita palabras no imprimibles (tenemos datos coreanos o bengalíes, por ejemplo)
        def is_latin(word):
            return all(char in string.printable for char in word)

        filtered_stop_words_multi = [word for word in stop_words_multi if is_latin(word)]

        # convierte 'title' en una matriz TF-IDF de dimensión (n_samples, n_features)
        self.vectorizer = TfidfVectorizer(stop_words=filtered_stop_words_multi)
        X_titles = self.vectorizer.fit_transform(df_recom['title'])

        # une todas las características en una única matriz dispersa, compuesta de vectores a partir de esas características
        X = sp.hstack((X_titles, 
                       sp.csr_matrix(df_recom['views_encoded'].values.reshape(-1, 1)), 
                       sp.csr_matrix(df_recom['likes_encoded'].values.reshape(-1, 1)), 
                       sp.csr_matrix(df_recom['dislikes_encoded'].values.reshape(-1, 1)), 
                       sp.csr_matrix(df_recom['comment_count_encoded'].values.reshape(-1, 1)), 
                       sp.csr_matrix(df_recom['category_encoded'].values.reshape(-1, 1)), 
                       sp.csr_matrix(df_recom['ratio_viralidad'].values.reshape(-1, 1)), 
                       sp.csr_matrix(df_recom['ratio_interaccion_dia_encoded'].values.reshape(-1, 1))))

        return df_recom, X

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
