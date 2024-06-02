import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from model import load_files, filter_columns, split_data, analyze_predictions
import sys
import keras_tuner as kt

def main():
    df = load_files(['samples F5.csv', 'samples F6.csv'], '-ENRICHED')


    X_train_full, X_test_full, y_train_full, y_test_full = split_data(df)



    X_train_prod = filter_columns(X_train_full, ['^NU-AP\d{5}$'])
    X_test_prod = filter_columns(X_test_full, ['^NU-AP\d{5}$'])
    y_train_prod = filter_columns(y_train_full, ['^NU-AP\d{5}$'])
    y_test_prod = filter_columns(y_test_full, ['^NU-AP\d{5}$'])

    if '--hyper-search' in sys.argv:
        search(X_train_full, X_test_full, y_train_full, y_test_full, X_train_prod, X_test_prod, y_train_prod, y_test_prod)
    elif '--run-hyper' in sys.argv:
        run_hyperparams(X_train_full, X_test_full, y_train_full, y_test_full, X_train_prod, X_test_prod, y_train_prod, y_test_prod)
    else:
        train(X_train_full, X_test_full, y_train_full, y_test_full, X_train_prod, X_test_prod, y_train_prod, y_test_prod)


def train(X_train_full, X_test_full, y_train_full, y_test_full, X_train_prod, X_test_prod, y_train_prod, y_test_prod):
    # Train the full feature model
    print(X_train_full.shape[1])
    full_model = build_full_model(X_train_full.shape[1])
    full_model.fit(X_train_full, y_train_full, epochs=100, validation_data=(X_test_full, y_test_full))

    # Build a new model for production that starts with similar architecture
    production_model = build_full_model(X_train_prod.shape[1])  # Note: same architecture, less input features

    # Initialize production model with weights from the full model (optional step for weight transfer)
    # production_model.set_weights(full_model.get_weights())  # Note: adjust the architecture if needed
    # for layer, full_layer in zip(production_model.layers[:], full_model.layers[:]):
        # print("Full model", full_layer.get_weights().shape, "prod model", layer.get_weights().shape)

    for layer, full_layer in zip(production_model.layers[2:-1], full_model.layers[2:-1]):
        layer.set_weights(full_layer.get_weights())

    # Now, retrain only the last layer(s) of the production model with the available production data
    # for layer in production_model.layers[:-1]:  # freeze all but the last layer
    #     layer.trainable = False

    production_model.compile(optimizer='adam', loss='mean_squared_error')
    production_model.fit(X_train_prod, y_train_prod, epochs=100, validation_data=(X_test_prod, y_test_prod))

    # Evaluate the standalone production model
    test_loss = production_model.evaluate(X_test_prod, y_test_prod)
    predictions = production_model.predict(X_test_prod)
    analyze_predictions(y_test_prod, predictions)
    print("Test loss with production features:", test_loss)

# Define the initial model structure
def build_full_model(input_shape):
    input_layer = layers.Input(shape=(input_shape,))
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(3)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def search(X_train_full, X_test_full, y_train_full, y_test_full, X_train_prod, X_test_prod, y_train_prod, y_test_prod):
    print(X_train_full.shape)
    hyperModel = MyHyperModel(X_train_full.shape)
    
    tuner = kt.BayesianOptimization(
        hyperModel,
        objective="mse",
        max_trials=3,
        executions_per_trial=3,
        directory="keras_hypermodels",
        project_name="project_name",
    )

    tuner.search(
        X_train_full, 
        y_train_full, 
        validation_split=0.2,
        validation_data=(X_test_full, y_test_full)
    )

    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Best hyperparameters: {best_hps.values}")

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X_train_full, y_train_full, epochs=100, validation_data=(X_test_full, y_test_full))

    test_loss = best_model.evaluate(X_test_full, y_test_full)
    predictions = best_model.predict(X_test_full)
    analyze_predictions(y_test_full, predictions)

HYPERPARAMS = {'normalization': False, 'num_layers': 5, 'units_0': 288, 'dropout': False, 'learning_rate': 0.001, 'units_1': 472, 'units_2': 208, 'units_3': 56, 'units_4': 504, 'dropout_rate': 0.05, 'batch_size': 80, 'epochs': 285, 'shuffle': True}
def run_hyperparams(X_train_full, X_test_full, y_train_full, y_test_full, X_train_prod, X_test_prod, y_train_prod, y_test_prod):
    
    full_model = MyHyperModel(X_train_full.shape).build_from_params(HYPERPARAMS)

    full_model.fit(X_train_full, y_train_full, epochs=HYPERPARAMS['epochs'], validation_data=(X_test_full, y_test_full))

    production_model = MyHyperModel(X_train_prod.shape).build_from_params(HYPERPARAMS)

    for layer, full_layer in zip(production_model.layers[2:-1], full_model.layers[2:-1]):
        layer.set_weights(full_layer.get_weights())

    production_model.compile(optimizer='adam', loss='mean_squared_error')
    production_model.fit(X_train_prod, y_train_prod, epochs=HYPERPARAMS['epochs'], validation_data=(X_test_prod, y_test_prod))

    test_loss = production_model.evaluate(X_test_prod, y_test_prod)
    predictions = production_model.predict(X_test_prod)
    analyze_predictions(y_test_prod, predictions)
    print("Test loss with production features:", test_loss)


    

MIN_EPOCHS = 5
MAX_EPOCHS = 305
STEP_EPOCHS = 10

MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 128
STEP_BATCH_SIZE = 8

MIN_LAYERS = 1
MAX_LAYERS = 5

MIN_UNITS = 8
MAX_UNITS = 512
STEP_UNITS = 8

MIN_DROPOUT = 0.0
MAX_DROPOUT = 0.5
STEP_DROPOUT = 0.05
DEFAULT_DROPOUT = 0.25

LEARNING_RATES = [1e-2, 1e-3, 1e-4]

class MyHyperModel(kt.HyperModel):
    def __init__(self, shape, name=None, tunable=True):
        super().__init__(name, tunable)
        self.shape=shape
    def build(self, hp):
        model = Sequential()
        model.add(Input((self.shape[1],)))
        # Whether to use normalization
        if hp.Boolean("normalization"):
            model.add(BatchNormalization())

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", MIN_LAYERS, MAX_LAYERS)):
            model.add(
                Dense(
                    # Tune number of units separately.
                    units=hp.Int(
                        f"units_{i}",
                        min_value=MIN_UNITS,
                        max_value=MAX_UNITS,
                        step=STEP_UNITS,
                    ),
                    activation="relu",
                )
            )
        if hp.Boolean("dropout"):
            model.add(
                Dropout(
                    rate=hp.Float(
                        "dropout_rate",
                        min_value=MIN_DROPOUT,
                        max_value=MAX_DROPOUT,
                        default=DEFAULT_DROPOUT,
                        step=STEP_DROPOUT,
                    )
                )
            )
        model.add(Dense(units=3, activation="linear"))

        # optimize learning rate
        hp_learning_rate = hp.Choice("learning_rate", values=LEARNING_RATES)
        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss="mean_squared_error",
            metrics=["mse", "mae"],
        )

        return model
    
    def build_from_params(self, params):
        model = Sequential()

        model.add(Input((self.shape[1],)))

        if params['normalization']:
            model.add(BatchNormalization())

        for i in range(params['num_layers']):
            model.add(
                Dense(
                    # Tune number of units separately.
                    units=params['units_'+ str(i)],
                    activation="relu",
                )
            )
        if params['dropout']:
            model.add(
                Dropout(
                    rate=params.dropout_rate
                )
            )
        model.add(Dense(units=3, activation="linear"))

        # optimize learning rate
        hp_learning_rate = params['learning_rate']

        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss="mean_squared_error",
            metrics=["mse", "mae"],
        )

        return model
            



    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
            batch_size=hp.Int(
                "batch_size",
                min_value=MIN_BATCH_SIZE,
                max_value=MAX_BATCH_SIZE,
                step=STEP_BATCH_SIZE,
            ),
            epochs=hp.Int(
                "epochs", min_value=MIN_EPOCHS, max_value=MAX_EPOCHS, step=STEP_EPOCHS
            ),
            shuffle=hp.Boolean("shuffle"),
        )




if __name__ == "__main__":
    main()