import pandas as pd
from model import split_data, VALIDATION_SPLIT, load_files, analyze_predictions, apply_pca
import keras_tuner as kt

import os
import sys

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

from neuralnet import MyHyperModel

LOAD_FLAG = '--load' in sys.argv

JSON = {'normalization': True, 'num_layers': 5, 'units_0': 136, 'dropout': True, 'learning_rate': 0.0001, 'units_1': 456, 'units_2': 488, 'units_3': 272, 'batch_size': 16, 'epochs': 235, 'shuffle': True, 'units_4': 272, 'dropout_rate': 0.45}

def main():
    df = load_files(['samples F5.csv', 'samples F6.csv'])

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

    if '--pca' in sys.argv:
        X_train, X_test = apply_pca(X_train, X_test)

    print(X_train)

    if LOAD_FLAG:
        model = load()
    else:
        model = train(X_train, y_train)


    predictions = model.predict(X_test)
    analyze_predictions(y_test, predictions)

def train(X_train, y_train):
    model = MyHyperModel(X_train.shape[1]).build(load_hyperparameters_from_json(JSON))

    model.fit(X_train, y_train, epochs=JSON['epochs'], validation_split=0.2)

    model.save(f"keras_hypermodels/'Trained/{find_newest_save()+1}.keras")

    return model

def load():
    name = sys.argv[sys.argv.index('--load') + 1]
    return load_model("keras_hypermodels/'Trained/"+name+".keras")

def find_newest_save():
    max = 0
    for file in os.listdir("keras_hypermodels/'Trained"):
        num = int(file.replace('.keras', ''))
        if num > max:
            max = num

    return max
        
    
    
def load_hyperparameters_from_json(params):
    
    hp = kt.HyperParameters()
    for param, value in params.items():
        if isinstance(value, bool):
            hp.Boolean(name=param, default=value)
        elif isinstance(value, int):
            hp.Int(name=param, default=value, min_value=value, max_value=value)
        elif isinstance(value, float):
            if param == 'learning_rate':
                hp.Choice(name=param, values=[value])
            else:
                hp.Float(name=param, default=value, min_value=value, max_value=value)
        elif isinstance(value, str):
            hp.Fixed(name=param, value=value)

    return hp


if __name__ == "__main__":
    main()