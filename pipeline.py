from datetime import datetime
import pickle
from util import load_files, filter_columns, evaluate_model, split_data, split_data_parts
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, ParameterGrid
import sys
import numpy as np
import pandas as pd

HYPERPARAM_SEARCH = '--hyper-search' in sys.argv
SAVE_MODEL = '--save' in sys.argv

class PipeLineModel(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict(X)
    
class SplitPipeline(Pipeline):
    def __init__(self, steps, parts, **kwargs):
        super().__init__(steps, **kwargs)

        self.parts = parts


    def fit(self, X, y):
        parts = split_data_parts(X, self.parts)

        X = parts[0]

        for (index, step) in enumerate(self.steps[:-1]):
            name, model = step

            model.fit(X, parts[index + 1])

            if index < len(self.steps) - 1:
                X = np.hstack((X, model.transform(X)))
        
        self.steps[-1][1].fit(X, y)
        
        return self
    
    def predict(self, X):
        parts = split_data_parts(X, self.parts)
        X = parts[0]


        for (index, step) in enumerate(self.steps):
            name, model = step

            if index < len(self.steps) - 1:
                X = np.hstack((X, model.transform(X)))
            else:
                return model.predict(X)
            
    def score(self, X, y):
        pred = self.predict(X)
        return -np.mean((y - pred) ** 2)

            

def main():
    df = load_files(["samplesF5-multilayer.csv", "samplesF6-multilayer.csv"])

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.5, random_state=0)

    # predict_location(X_train, X_test, y_train, y_test)
    distance_to_location(X_train, X_test, y_train, y_test)



def predict_location(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=175, #175 1870
        max_depth=120, #120 None
        min_samples_split=4, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    search_grid = {
        'location__n_estimators': [50, 100],
        'location__max_depth': [None, 10],
    }

    pipeline = Pipeline([
        ('location', model)
    ])

    handle_pipeline(pipeline, "Direct location prediction", X_train, X_test, y_train, y_test, search_grid=search_grid)
    
def distance_to_location(X_train, X_test, y_train, y_test):
    distance_model = RandomForestRegressor(
        n_estimators=1500, #175 1870
        max_depth=None, #120 None
        min_samples_split=2, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    location_model = RandomForestRegressor(
        n_estimators=500, #175 1870
        max_depth=None, #120 None
        min_samples_split=2, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    search_grid = {
        'distance__model__n_estimators': [100, 1500, 2000],
        'distance__model__min_samples_split': [2],
        'distance__model__min_samples_leaf': [1],
        'location__n_estimators': [100, 500, 1500],
        'location__min_samples_split': [2],
        'location__min_samples_leaf': [1],
    }

    pipeline = SplitPipeline([
        ('distance', PipeLineModel(distance_model)),
        ('location', location_model)
    ], [['^NU-AP\d{5}$'], ['^NU-AP\d{5}_distance$']])


    handle_pipeline(pipeline, "Distance-to-location", X_train, X_test, y_train, y_test, search_grid=search_grid)

def handle_pipeline(pipeline, name, X_train, X_test, y_train, y_test, search_grid):

    print(f"\n\n##### {name} ####")

    if HYPERPARAM_SEARCH:
        grid_search = GridSearchCV(pipeline, search_grid, n_jobs=3, verbose=3, error_score='raise', pre_dispatch=3)
        grid_search.fit(X_train, y_train)

        # Output the best parameters and score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print("Test set score: ", test_score)
    else:
        fitted_model = pipeline.fit(X_train, y_train)

        predictions = fitted_model.predict(X_test)

        if type(y_test) is list:
            y_test = y_test[-1]

        evaluate_model(y_test, predictions, name)

        if(SAVE_MODEL):
            now = datetime.now().strftime('%m-%d--%H-%M-%S')

            pickle.dump(fitted_model, open(f"sklearn_models/multilayer/{name}-{now}.pkl", 'wb'))
        

        



if __name__ == '__main__':
    main()