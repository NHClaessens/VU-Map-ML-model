from util import load_files, filter_tuple, evaluate_model, split_data_parts
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, ParameterGrid
import sys
import numpy as np

HYPERPARAM_SEARCH = '--hyper-search' in sys.argv

class PipeLineModel(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict(X)
    

def main():
    df = load_files(["samplesF5-multilayer.csv", "samplesF6-multilayer.csv"])

    data_parts = split_data_parts(df, regex_list=[['^NU-AP\d{5}$'], ['^NU-AP\d{5}_distance$']])



    # predict_location(data_parts['part_1_train'], data_parts['part_1_test'], data_parts['y_train'], data_parts['y_test'])
    distance_to_location(data_parts['part_1_train'], data_parts['part_1_test'], data_parts['part_2_train'], data_parts['part_2_test'], data_parts['y_train'], data_parts['y_test'])



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
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
    }

    handle_model(model, "Direct location prediction", X_train, X_test, y_train, y_test, search_grid=search_grid)


class DistanceToLocationRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, distance_model=RandomForestRegressor(), location_model=RandomForestRegressor()):
        self.distance_model = distance_model
        self.location_model = location_model

    def fit(self, X, y):
        # Assume y is a tuple (y_distances, y_locations)
        y = filter_tuple(y, None)
        print(len(y), y)
        y_distances, y_locations = y

        # Fit distance model
        self.distance_model.fit(X, y_distances)

        # Predict distances and combine with original features
        distances_pred = self.distance_model.predict(X)
        X_combined = np.hstack((X, distances_pred))

        # Fit location model
        self.location_model.fit(X_combined, y_locations)
        
        return self

    def predict(self, X):
        # Predict distances and combine with original features
        distances_pred = self.distance_model.predict(X)
        X_combined = np.hstack((X, distances_pred))        

        # Predict locations
        return self.location_model.predict(X_combined)
    
    def score(self, X, y):
        # Custom scoring method for compatibility with GridSearchCV
        y = filter_tuple(y, None)
        _, y_locations = y
        y_pred = self.predict(X)
        return -np.mean((y_locations - y_pred) ** 2)  # Negative MSE for GridSearchCV to maximize
    
def distance_to_location(RSSI_train, RSSI_test, dist_train, dist_test, loc_train, loc_test):
    distance_model = RandomForestRegressor(
        n_estimators=1000, #175 1870
        max_depth=None, #120 None
        min_samples_split=2, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    location_model = RandomForestRegressor(
        n_estimators=1000, #175 1870
        max_depth=None, #120 None
        min_samples_split=2, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    model = DistanceToLocationRegressor(distance_model, location_model)

    search_grid = {
        'distance_model__n_estimators': [100, 1500, 2000],
        'distance_model__min_samples_split': [2],
        'distance_model__min_samples_leaf': [1],
        'location_model__n_estimators': [100, 500, 1500],
        'location_model__min_samples_split': [2],
        'location_model__min_samples_leaf': [1],
    }

    handle_model(model, "Distance -> location", RSSI_train, RSSI_test, (dist_train, loc_train), (dist_test, loc_test), search_grid=search_grid)

def handle_model(model, name, X_train, X_test, y_train, y_test, search_grid):

    print(f"\n\n##### {name} ####")

    if HYPERPARAM_SEARCH:
        grid_search = GridSearchCV(model, search_grid, n_jobs=-1, verbose=3)
        y_train += (None,) * (len(X_train) - len(y_train))
        grid_search.fit(X_train, y_train)

        # Output the best parameters and score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print("Test set score: ", test_score)
    else:
        fitted_model = model.fit(X_train, y_train)

        predictions = fitted_model.predict(X_test)

        if type(y_test) is tuple:
            y_test = y_test[-1]

        evaluate_model(y_test, predictions, name)
        

        



if __name__ == '__main__':
    main()