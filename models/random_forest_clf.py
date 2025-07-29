"""
Modelo Random Forest Classifier de Sklearn

Clase: RandomForestClf
"""

import numpy as np
import pickle
import configparser

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from utils.data_utils import evaluar_modelo_train_test, evaluar_modelo_y_pred, balancear_pesos

# fichero train.conf
FILE_CONFIG = 'config/train.conf'
SECCION_CONFIG = 'RandomForest_'
SECCION_FILE_TRAINED = 'FilesTrained'
FILE_TRAINED = 'random_forest_'

class RandomForestClf:
    """
    Random Forest Classifier
    Métodos:
    fit: Entrenamiento del modelo
    predict: Predicciones a nuevos datos a partir del modelo ya entrenado
    save: Guarda en disco serializado en pickle el objeto RandomForestClf actual
    load: Carga de disco un objeto RandomForestClf con el entrenamiento del modelo
    __preprocess_data: Preprocesamiento del dataset antes de aplicar el modelo
    __obtener_nombre_archivo: Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
    """

    def __init__(self):

        # inicializa el modelo
        self.model = None

    def fit(self, df, variable):
        '''
        Entrenamiento del modelo
        :df: dataframe de Pandas con el dataset
        :variable: variable objetivo
        '''

        # fichero de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

        # preprocesamiento de datos
        X, y = self.__preprocess_data(df, variable)

        # conjuntos de entrenamiento y pruebas
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # balancea la variable objetivo
        class_weight = {}
        match variable:
            case 'category':
                class_weight = balancear_pesos(y_train, 'category_encoded')
            case 'comments':
                class_weight = balancear_pesos(y_train, 'comments_disabled')
            case 'ratings':
                class_weight = balancear_pesos(y_train, 'ratings_disabled')

        # parámetros del modelo

        # bootstrap: bool
        bootstrap = config.getboolean(SECCION_CONFIG + variable, 'bootstrap')
        # ccp_alpha: float
        ccp_alpha = config.getfloat(SECCION_CONFIG + variable, 'ccp_alpha')
        # criterion: str
        criterion = config.get(SECCION_CONFIG + variable, 'criterion')
        # max_depth: None, int
        max_depth = config.get(SECCION_CONFIG + variable, 'max_depth')
        if max_depth == 'None':
            max_depth = None
        else:
            max_depth = config.getint(SECCION_CONFIG + variable, 'max_depth')
        # max_features: None, int, float, str
        max_features = config.get(SECCION_CONFIG + variable, 'max_features')
        if max_features == 'None':
            max_features = None
        else:
            try:
                # intenta convertirlo a int
                max_features = int(max_features)
            except ValueError:
                pass
            try:
                # intenta convertirlo a float
                max_features = float(max_features)
            except ValueError:
                pass
            # si no es un número, lo deja como string
            max_features = max_features
        # max_leaf_nodes: None, int [2, inf)
        max_leaf_nodes = config.get(SECCION_CONFIG + variable, 'max_leaf_nodes')
        if max_leaf_nodes == 'None':
            max_leaf_nodes = None
        else:
            max_leaf_nodes = config.getint(SECCION_CONFIG + variable, 'max_leaf_nodes')
        # min_impurity_decrease: float
        min_impurity_decrease = config.getfloat(SECCION_CONFIG + variable, 'min_impurity_decrease')
        # min_samples_leaf: int [1, inf); float (0.0, 1.0)
        min_samples_leaf = config.getfloat(SECCION_CONFIG + variable, 'min_samples_leaf')
        if min_samples_leaf <= 0 or min_samples_leaf >= 1:
            min_samples_leaf = int(min_samples_leaf)
        # min_samples_split: int [2, inf); float (0.0, 1.0]
        min_samples_split = config.getfloat(SECCION_CONFIG + variable, 'min_samples_split')
        if min_samples_split > 1:
            min_samples_split = int(min_samples_split)
        # n_estimators: int
        n_estimators = config.getint(SECCION_CONFIG + variable, 'n_estimators')
        # random_state: int
        random_state = config.getint(SECCION_CONFIG + variable, 'random_state')

        params_rf = {'bootstrap': bootstrap, 'ccp_alpha': ccp_alpha, 'class_weight': class_weight,
                     'criterion': criterion, 'max_depth': max_depth, 'max_features': max_features,
                     'max_leaf_nodes': max_leaf_nodes, 'min_impurity_decrease': min_impurity_decrease,
                     'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
                     'n_estimators': n_estimators, 'random_state': random_state}

        # instancia Random Forest
        self.model = RandomForestClassifier(**params_rf)

        # entrena el modelo
        self.model.fit(X_train, np.ravel(y_train))

        # evalúa el modelo
        evaluar_modelo_train_test(self.model, X_train, y_train, X_test, y_test)

    def predict(self, df, variable):
        '''
        Predicciones a nuevos datos a partir del modelo ya entrenado.
        :df: DataFrame con los nuevos datos
        :variable: variable objetivo
        :return: predicciones
        '''

        # verifica que el modelo esté cargado
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado. Use el método 'load' para cargarlo.")

        # preprocesa los datos recibidos
        X, y = self.__preprocess_data(df, variable)

        # realiza la predicción utilizando el modelo cargado
        predictions = self.model.predict(X)

        # evalúa la predicción
        evaluar_modelo_y_pred(self.model, y, predictions)

        # devuelve la predicción
        return predictions

    def save(self, directory_save, variable):
        '''
        Guarda en disco serializado en pickle el objeto RandomForestClf actual
        :directory_save: directorio de los datos ya entrenados
        :variable: variable objetivo
        '''

        # obtiene el nombre del archivo a guardar
        path_file = directory_save + self.__obtener_nombre_archivo(variable)

        # abre archivo para escritura
        try:
            with open(path_file, 'wb') as fp:
                pickle.dump(self, fp)
            print('Datos guardados en el fichero:', path_file)
        except IOError as e:
            raise IOError('Error al intentar guardar los datos en el archivo' + str(path_file) + ':', e)

    def load(self, directory_load, variable):
        '''
        Carga de disco un objeto RandomForestClf con el entrenamiento del modelo
        :directory_load: directorio de los datos ya entrenados
        :variable: variable objetivo
        '''

        # obtiene el nombre del archivo a leer
        path_file = directory_load + self.__obtener_nombre_archivo(variable)

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

    def __preprocess_data(self, df, variable):
        '''
        Preprocesamiento del dataset antes de aplicar el modelo
        :df: DataFrame de Pandas
        :variable: variable objetivo
        :return: X (variables de entrada), y (variable de salida)
        '''

        match variable:
            case 'category':

                # selecciona las variables de entrada de entre las columnas del dataset
                X = df[['channel_title_encoded', 'tags_encoded', 'views_encoded', 'likes_encoded', 'dislikes_encoded',
                        'comment_count_encoded', 'comments_disabled', 'ratings_disabled', 'country_encoded', 'ratio_viralidad',
                        'ratio_interaccion_dia_encoded', 'trending_date_year_encoded', 'trending_date_month_encoded',
                        'publish_time_year_encoded', 'publish_time_month_encoded', 'publish_trending_days']]

                # selecciona la variable de salida
                y = df[['category_encoded']].copy(deep=True)

                # elimina las 7 características de menor peso
                X = X.drop(['ratings_disabled', 'trending_date_year_encoded', 'publish_time_year_encoded',
                            'comments_disabled', 'trending_date_month_encoded', 'publish_time_month_encoded',
                            'publish_trending_days'], axis='columns')

            case 'comments':

                # selecciona las variables de entrada de entre las columnas del dataset
                X = df[['channel_title_encoded', 'tags_encoded', 'views_encoded', 'likes_encoded', 'dislikes_encoded',
                        'comment_count_encoded', 'ratings_disabled', 'category_encoded', 'country_encoded',
                        'ratio_viralidad', 'ratio_interaccion_dia_encoded', 'trending_date_year_encoded',
                        'trending_date_month_encoded', 'publish_time_year_encoded', 'publish_time_month_encoded',
                        'publish_trending_days']]

                # selecciona la variable de salida
                y = df[['comments_disabled']].copy(deep=True)

            case 'ratings':

                # selecciona las variables de entrada de entre las columnas del dataset
                X = df[['channel_title_encoded', 'tags_encoded', 'views_encoded', 'likes_encoded', 'dislikes_encoded',
                        'comment_count_encoded', 'comments_disabled', 'category_encoded', 'country_encoded',
                        'ratio_viralidad', 'ratio_interaccion_dia_encoded', 'trending_date_year_encoded',
                        'trending_date_month_encoded', 'publish_time_year_encoded', 'publish_time_month_encoded',
                        'publish_trending_days']]

                # selecciona la variable de salida
                y = df[['ratings_disabled']].copy(deep=True)

        # devuelve ambas variables
        return X, y

    def __obtener_nombre_archivo(self, variable):
        '''
        Obtiene del archivo de configuración el nombre del archivo donde se guarda este modelo
        :variable: variable objetivo
        :return: nombre del archivo (sin directorios)
        '''

        # archivo de configuración
        config = configparser.ConfigParser()
        config.read(FILE_CONFIG)

        # lee y devuelve el nombre del archivo
        return config[SECCION_FILE_TRAINED][FILE_TRAINED + variable]
