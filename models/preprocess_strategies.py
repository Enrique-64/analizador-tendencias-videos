"""
Familia de algoritmos para preprocesar los datos.
- StandardScaler
- MinMaxScaler
- LabelEncoder

Ejemplo de utilización:

# variables para StandardScaler
preprocessing_strategy = AplicarStandardScaler()

# var1
data = df_encoded['var1'].to_numpy().reshape(-1, 1)
model_var1, df_encoded['var1_encoded'] = preprocessing_strategy.preprocess(data)
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class PreprocessingStrategy:
    def preprocess(self, data):
        raise NotImplementedError('Error: método "preprocess" no implementado.')

class AplicarStandardScaler(PreprocessingStrategy):
    """
    StandardScaler de Scikit-Learn
    """
    def preprocess(self, data):
        scaler = StandardScaler()
        model_scaler = scaler.fit(data)
        data_scaler = model_scaler.transform(data)
        return model_scaler, data_scaler

class AplicarMinMaxScaler(PreprocessingStrategy):
    """
    MinMaxScaler de Scikit-Learn
    """
    def preprocess(self, data):
        scaler = MinMaxScaler()
        model_scaler = scaler.fit(data)
        data_scaler = model_scaler.transform(data)
        return model_scaler, data_scaler

class AplicarLabelEncoder(PreprocessingStrategy):
    """
    LabelEncoder de Scikit-Learn
    """
    def preprocess(self, data):
        label = LabelEncoder()
        model_label = label.fit(data)
        data_label = model_label.transform(data)
        return model_label, data_label
