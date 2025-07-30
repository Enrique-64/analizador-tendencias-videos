## Analizador de tendencias de vídeos de Internet

## Descripción

Este proyecto aplica técnicas de Machine Learning con Python para analizar tendencias diarias de vídeos de YouTube. Utiliza múltiples modelos para resolver tareas de clasificación, regresión, agrupación y recomendación.

## Objetivos del proyecto

1. Clasificar la categoría del vídeo.
2. Predecir si los comentarios estarán deshabilitados.
3. Predecir si los ratings estarán deshabilitados.
4. Predecir el número de "likes".
5. Estimar la ratio "likes/dislikes".
6. Agrupar vídeos con técnicas de clustering.
7. Recomendar vídeos basados en sus características.

Para cada uno de los objetivos, se realiza un análisis exploratorio de los datos seguido de la evaluación de distintas técnicas y modelos de Machine Learning. Se seleccionan aquellos que ofrecen el mejor rendimiento y se implementa un script de Python específico para cada modelo elegido.

## Estructura del repositorio

.  
├── config/ # Configuración e hiperparámetros
├── data/
│ ├── raw/ # Datos originales sin procesar
│ └── processed/ # Datos ya transformados
├── models/ # Código para cada modelo de ML
├── notebooks/ # Análisis exploratorio y modelado en Jupyter Lab
├── utils/ # Funciones auxiliares reutilizables
├── inferencie_model.py # Realiza inferencias con modelos entrenados
├── preprocess_data.py # Preprocesa los datos originales
├── train_models.py # Entrena los modelos definidos
└── README.md # Descripción del proyecto


## Requisitos

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyterlab

Instala los requisitos con:

```bash
pip install -r requirements.txt
```

## Ejecución

1. Preprocesamiento de datos:

```bash
python preprocess_data.py
```

2. Entrenamiento de modelos:

```bash
python train_models.py
```

3. Inferencia:

```bash
python inferencie_model.py
```

También puedes explorar los notebooks en la carpeta notebooks/ para visualizar los análisis y modelos de forma interactiva.

## Autor

**Enrique** - [listas.emo@gmail.com] - GitHub

Licencia MIT © 2024 Enrique