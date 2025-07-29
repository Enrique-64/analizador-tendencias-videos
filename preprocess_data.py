"""
Carga de datos iniciales.
Transformación, limpieza y procesado de datos, comunes a todos los modelos.

1) Ejecución desde el terminal:

Los parámetros necesarios se recogen del archivo de configuración recibido como argumento a través del terminal:

--path_original: Ruta "en crudo" del directorio de los datos originales (ej.: "C:\\Python\\raw")
--path_processed: Ruta del directorio donde dejar los datos procesados (ej.: "data\\processed")

2) Ejecución desde fuera del terminal:

Los parámetros necesarios se recogen del archivo 'train.conf' situado en el directorio 'config'
"""

import pandas as pd

from datetime import datetime

import configparser
import argparse
import sys

from models.preprocess_strategies import AplicarStandardScaler, AplicarMinMaxScaler, AplicarLabelEncoder

SECCION_CONFIG = 'DEFAULT'
PATH_LOAD_DATOS = 'path_original'
PATH_SAVE_DATOS = 'path_processed'

if __name__ == '__main__':

    # carga el fichero de configuración
    if sys.stdin.isatty():
        # el script sí se está ejecutando desde el terminal

        # llamada desde el terminal
        # python inference_model.py --config_path models/saved_models/model_1.pkl --data_path data/processed/new_data.csv

        parser = argparse.ArgumentParser(description="Tratamiento de datos originales")

        # argumentos que el usuario pasa al script
        parser.add_argument('--path_original', required=True,
                            help=r'Ruta "en crudo" del directorio de los datos originales (ej.: "C:\Python\raw")')
        parser.add_argument('--path_processed', required=True,
                            help=r'Ruta del directorio donde dejar los datos procesados (ej.: "data\processed")')
        args = parser.parse_args()

        # directorio donde se encuentran los ficheros originales de datos
        directory_load = args.path_original + '\\'

        # directorio donde se guardarán los datos preprocesados
        directory_save = args.path_processed + '\\'

    else:
        # el script se está ejecutando desde fuera del terminal

        # accede al fichero de configuración
        config = configparser.ConfigParser()
        config.read('config/train.conf')

        # directorio donde se encuentran los ficheros originales de datos
        directory_load = config[SECCION_CONFIG][PATH_LOAD_DATOS]

        # directorio donde se guardarán los datos preprocesados
        directory_save = config[SECCION_CONFIG][PATH_SAVE_DATOS]

    '''
    Estudio de los datos
    '''

    print('Iniciando carga y procesamiento de datos originales...')

    # Creamos dos diccionarios, en uno se almacenarán los ficheros csv y en otro los json.
    csv = {}
    json_dict = {}
    file_path = ''

    try:
        file_path = directory_load + 'CAvideos.csv'
        csv['CA'] = pd.read_csv(file_path)

        file_path = directory_load + 'DEvideos.csv'
        csv['DE'] = pd.read_csv(file_path)

        file_path = directory_load + 'FRvideos.csv'
        csv['FR'] = pd.read_csv(file_path)

        file_path = directory_load + 'GBvideos.csv'
        csv['GB'] = pd.read_csv(file_path)

        file_path = directory_load + 'INvideos.csv'
        csv['IN'] = pd.read_csv(file_path)

        file_path = directory_load + 'USvideos.csv'
        csv['US'] = pd.read_csv(file_path)

        # algunos de los ficheros no tienen codificación UTF-8:
        file_path = directory_load + 'JPvideos.csv'
        csv['JP'] = pd.read_csv(file_path, encoding='latin-1')

        file_path = directory_load + 'KRvideos.csv'
        csv['KR'] = pd.read_csv(file_path, encoding='latin-1')

        file_path = directory_load + 'MXvideos.csv'
        csv['MX'] = pd.read_csv(file_path, encoding='latin-1')

        file_path = directory_load + 'RUvideos.csv'
        csv['RU'] = pd.read_csv(file_path, encoding='latin-1')

        # Ficheros json
        file_path = directory_load + 'CA_category_id.json'
        json_dict['CA'] = pd.read_json(file_path)

        file_path = directory_load + 'DE_category_id.json'
        json_dict['DE'] = pd.read_json(file_path)

        file_path = directory_load + 'FR_category_id.json'
        json_dict['FR'] = pd.read_json(file_path)

        file_path = directory_load + 'GB_category_id.json'
        json_dict['GB'] = pd.read_json(file_path)

        file_path = directory_load + 'IN_category_id.json'
        json_dict['IN'] = pd.read_json(file_path)

        file_path = directory_load + 'JP_category_id.json'
        json_dict['JP'] = pd.read_json(file_path)

        file_path = directory_load + 'KR_category_id.json'
        json_dict['KR'] = pd.read_json(file_path)

        file_path = directory_load + 'MX_category_id.json'
        json_dict['MX'] = pd.read_json(file_path)

        file_path = directory_load + 'RU_category_id.json'
        json_dict['RU'] = pd.read_json(file_path)

        file_path = directory_load + 'US_category_id.json'
        json_dict['US'] = pd.read_json(file_path)

    except IOError as e:
        raise IOError('Error al intentar abrir el fichero ' + file_path + ': ', e)

    '''
    Unificación de los datos: recoger la información de categorías del fichero json de cada país y a añadirla al fichero 
    csv de dicho país
    '''

    print('Unificando datos...')

    # guarda en dict_cats todas las categorías de cada país
    # dict_cats: { cod_pais: {id_cat: nombre_cat}, ... }
    country_list = ['CA', 'DE', 'FR', 'GB', 'IN', 'JP', 'KR', 'MX', 'RU', 'US']
    dict_cats = {}
    for country in country_list:
        dict = {}
        for j in json_dict[country]['items']:
            if j['id'] not in dict:
                dict[j['id']] = j['snippet']['title']
        dict_cats[country] = dict

    # añade en csv_category una columna con el título de la categoría y otra con el código de país
    csv_category = csv.copy()

    for country in country_list:
        lista_titulos = []
        # recorre todas las filas del fichero csv
        for fila in range(0, csv_category[country].shape[0]):
            id = csv_category[country].loc[fila]['category_id']
            titulo = ''
            # busca el id de categoría en el fichero json
            for j in json_dict[country]['items']:
                if int(j['id']) == id:
                    # categoría encontrada en el fichero json
                    titulo = j['snippet']['title']
                    break

            if titulo == '':
                titulo = 'categoría no encontrada en el fichero json'
            # guarda el nombre de la categoría
            lista_titulos.append(titulo)

        # añadimos la lista de nombres de categorías al dataframe
        csv_category[country]['category'] = lista_titulos

        # añadimos el código de país al dataframe
        csv_category[country]['country'] = country

    # creamos un único dataframe que combine los dataframes de todos los países

    print('Creando nuevo dataset completo...')

    # recorremos los países, concatenando los csv uno por uno al primero

    df = pd.DataFrame

    kont = 1
    for country in country_list:
        if kont == 1:
            df = csv_category[country].copy()
        else:
            df = pd.concat([df, csv_category[country]], ignore_index=True)
        kont += 1

    print('Guardando dataset en disco...')

    # guardamos los datos en disco
    file_path = directory_save + 'csv_completo.csv'

    try:
        df.to_csv(file_path, index=False)
    except IOError as e:
        raise IOError('Error al intentar guardar el fichero ' + file_path + ': ', e)

    '''
    EDA
    '''

    print('Iniciando EDA...')

    # eliminamos variables que no vamos a necesitar en el estudio
    df.drop(['video_id', 'category_id', 'thumbnail_link', 'video_error_or_removed'], axis='columns', inplace=True)

    # eliminamos filas duplicadas
    df.drop_duplicates(inplace=True)

    # trending_date: Campo de fecha con formato 'yy.dd.mm', vamos a convertirlo en datetime.
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
    # separamos el año y el mes de la fecha de tendencia
    df['trending_date_str'] = df['trending_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    df['trending_date_year'] = df['trending_date_str'].apply(lambda x: datetime.strptime(x, ("%Y-%m-%d")).year)
    df['trending_date_month'] = df['trending_date_str'].apply(lambda x: datetime.strptime(x, ("%Y-%m-%d")).month)

    # publish_time: Campo de fecha, vamos a convertirlo en datetime.
    df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # separamos el año y el mes de la fecha de publicación
    df['publish_time_str'] = df['publish_time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    df['publish_time_year'] = df['publish_time_str'].apply(lambda x: datetime.strptime(x, ("%Y-%m-%d")).year)
    df['publish_time_month'] = df['publish_time_str'].apply(lambda x: datetime.strptime(x, ("%Y-%m-%d")).month)

    # Ratio de interacción total o de viralidad
    df['ratio_viralidad'] = (df.apply
                             (lambda row: ( row['likes'] + row['dislikes'] + row['comment_count'] ) / row['views']
                             if row['views'] != 0 else None, axis=1))

    # Ratio de interacción por día
    df['ratio_interaccion_dia'] = (df.apply
                                   (lambda row: ( row['likes'] + row['dislikes'] + row['comment_count'] ) /
                                                (datetime.now() - row['publish_time']).days
                                   if (datetime.now() - row['publish_time']).days != 0 else None, axis=1))

    # Eliminación de otras variables
    df.drop(['description'], axis='columns', inplace=True)

    # publish_trending_days: Nueva variable: días transcurridos entre la fecha de publicación y la fecha de tendencia.
    df['publish_trending_days'] = df.apply(lambda row: (row['trending_date'] - row['publish_time']).days, axis=1)

    # Eliminación de un 0,3% de registros con fechas erróneas
    valor_inferior = 0.05
    valor_superior = 0.997
    percentil_inferior = df['publish_trending_days'].quantile(valor_inferior)
    percentil_superior = df['publish_trending_days'].quantile(valor_superior)
    df = df[(df['publish_trending_days'] >= percentil_inferior) & (df['publish_trending_days'] <= percentil_superior)]

    '''
    Codificación y escalado de datos
    '''

    print('Codificado y escalado de datos...')

    df_encoded = df.copy()

    # Codificación de variables categóricas

    # variables para LabelEncoder
    preprocessing_strategy = AplicarLabelEncoder()

    # channel_title
    data = df_encoded['channel_title']
    model_channel_title, df_encoded['channel_title_encoded'] = preprocessing_strategy.preprocess(data)

    # tags
    data = df_encoded['tags']
    model_tags, df_encoded['tags_encoded'] = preprocessing_strategy.preprocess(data)

    # category
    data = df_encoded['category']
    model_category, df_encoded['category_encoded'] = preprocessing_strategy.preprocess(data)

    # country
    data = df_encoded['country']
    model_country, df_encoded['country_encoded'] = preprocessing_strategy.preprocess(data)

    # Escalado de variables numéricas

    # variables para StandardScaler
    preprocessing_strategy = AplicarStandardScaler()

    # views
    data = df_encoded['views'].to_numpy().reshape(-1, 1)
    model_views, df_encoded['views_encoded'] = preprocessing_strategy.preprocess(data)

    # likes
    data = df_encoded['likes'].to_numpy().reshape(-1, 1)
    model_likes, df_encoded['likes_encoded'] = preprocessing_strategy.preprocess(data)

    # dislikes
    data = df_encoded['dislikes'].to_numpy().reshape(-1, 1)
    model_dislikes, df_encoded['dislikes_encoded'] = preprocessing_strategy.preprocess(data)

    # comment_count
    data = df_encoded['comment_count'].to_numpy().reshape(-1, 1)
    model_comment_count, df_encoded['comment_count_encoded'] = preprocessing_strategy.preprocess(data)

    # ratio_interaccion_dia
    data = df_encoded['ratio_interaccion_dia'].to_numpy().reshape(-1, 1)
    model_ratio_interaccion_dia, df_encoded['ratio_interaccion_dia_encoded'] = preprocessing_strategy.preprocess(data)

    # trending_date_year
    data = df_encoded['trending_date_year'].to_numpy().reshape(-1, 1)
    model_trending_date_year, df_encoded['trending_date_year_encoded'] = preprocessing_strategy.preprocess(data)

    # publish_time_year
    data = df_encoded['publish_time_year'].to_numpy().reshape(-1, 1)
    model_publish_time_year, df_encoded['publish_time_year_encoded'] = preprocessing_strategy.preprocess(data)

    # variables para MinMaxScaler
    preprocessing_strategy = AplicarMinMaxScaler()

    # trending_date_month
    data = df_encoded['trending_date_month'].to_numpy().reshape(-1, 1)
    (model_trending_date_month, df_encoded['trending_date_month_encoded']) = preprocessing_strategy.preprocess(data)

    # publish_time_month
    data = df_encoded['publish_time_month'].to_numpy().reshape(-1, 1)
    model_publish_time_month, df_encoded['publish_time_month_encoded'] = preprocessing_strategy.preprocess(data)

    # Guardar el disco el dataset preparado para los modelos

    # Eliminamos las variables que no van a ser de utilidad
    df_encoded.drop(['trending_date', 'publish_time', 'trending_date_str', 'publish_time_str'], axis='columns',
                    inplace=True)

    print('Guardando dataset en disco...')

    # Guardamos en disco

    file_path = directory_save + 'csv_eda.csv'

    try:
        df_encoded.to_csv(file_path)
    except IOError as e:
        raise IOError('Error al intentar guardar el fichero ' + file_path + ': ', e)

print('Acabado')
