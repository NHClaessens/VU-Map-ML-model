import pandas as pd
from model import split_data, VALIDATION_SPLIT, load_files
import keras_tuner as kt

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


def main():
    df = load_files(['samples F5.csv', 'samples F6.csv'])

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

    print(X_train.shape)

    project_name = "TEST3"

    # keras tuner
    hyperModel = MyHyperModel(X_train.shape)
    tuner = kt.BayesianOptimization(
        hyperModel,
        objective="mse",
        max_trials=3,
        executions_per_trial=3,
        directory="keras_hypermodels",
        project_name=project_name,
    )

    # search for best hyperparameters
    tuner.search(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        validation_data=(X_test, y_test),
    )

    best_model = tuner.get_best_models()[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0]

    best_model.build(X_train.shape[1:])

    print("Best Model:")
    print("----------------------------------------------")
    print(best_model.summary())
    print("----------------------------------------------")
    print("Best Hyperparameters:")
    print("----------------------------------------------")
    print(best_hyperparameters.values)
    print("----------------------------------------------")

    # best_model.predict()


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
        self.shape=shape[1]
    def build(self, hp):
        model = Sequential()
        model.add(Flatten(input_shape=(self.shape,)))
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