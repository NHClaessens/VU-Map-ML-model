from datetime import datetime
import json
import pickle

from sklearn.discriminant_analysis import StandardScaler
from util import get_ap_locations, load_files, filter_columns, evaluate_model, split_data, split_data_parts, triangulate
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, ParameterGrid
import sys
import numpy as np
import pandas as pd

HYPERPARAM_SEARCH = '--hyper-search' in sys.argv
SAVE_MODEL = '--save' in sys.argv
SOCKET = '--socket' in sys.argv

unity = None

if(SOCKET):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 65432))
            s.listen()
            print('Waiting for Unity to connect')
            conn, addr = s.accept()
            print(f"Connected at {addr}")
            unity = conn

ap_positions = None

class PipeLineModel(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict(X)
    
class SplitPipeline(Pipeline):
    def __init__(self, steps, start, targets, type='normal', **kwargs):
        super().__init__(steps, **kwargs)

        self.start = start
        self.targets = targets
        self.type = type


    def fit(self, X, y):
        parts = split_data_parts(X, [self.start, *self.targets])
        X = parts[0]
        targets = parts[1:]

        for (index, step) in enumerate(self.steps[:-1]):
            name, model = step

            model.fit(X, targets[index])

            if index < len(self.steps) - 1:
                if self.type == 'cumulative':
                    X = np.hstack((X, model.transform(X)))
                if self.type == 'normal':
                    X = model.transform(X)
            
        self.steps[-1][1].fit(X, y)
        
        return self
    
    def predict(self, X):
        X = split_data_parts(X, [self.start])[0]

        for (index, step) in enumerate(self.steps):
            name, model = step

            if index < len(self.steps) - 1:
                if self.type == 'cumulative':
                    X = np.hstack((X, model.transform(X)))
                if self.type == 'normal':
                    X = model.transform(X)
            else:
                return model.predict(X)
            
    def score(self, X, y):
        pred = self.predict(X)
        return -np.mean((y - pred) ** 2)
    
    def evaluate(self, X, y):
        parts = split_data_parts(X, [self.start, *self.targets])
        X = parts[0]
        targets = parts[1:]

        for (index, step) in enumerate(self.steps):
            name, model = step

            if index < len(self.steps) - 1:
                predictions = model.transform(X)

                evaluate_model(targets[index], predictions, f'Layer {index + 1}: {name}')

                if self.type == 'cumulative':
                    X = np.hstack((X, predictions))
                if self.type == 'normal':
                    X = predictions
            else:
                predictions = model.predict(X)
                evaluate_model(y, predictions, f'{index}: {name}', location=True)

                return predictions

class TriangulationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res =  np.apply_along_axis(self.triangulate, 1, X)
        return res
    
    def predict(self, X):
        return self.transform(X)
    
    def triangulate(self, distances):
        return triangulate(distances, ap_positions)

class ObstacleTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        if(unity is None):
            print("Please enable socket with --socket")
            exit(-1)

    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return self.transform(X)
    
    def transform(self, X):
        res =  np.apply_along_axis(self.get_obstacles, 1, X)
        return res


    def get_obstacles(self, pos):
        unity.sendall(json.dumps({
            "type": "obstacles",
            "data": {
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
            }
        }).encode())
        
        data = unity.recv(10000).decode()

        data = json.loads(data)
        obstacles = json.loads(data['data']['obstacle_thickness'])
        return obstacles

def main():
    df = load_files(["samplesF5-multilayer.csv", "samplesF6-multilayer.csv"])

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.5, random_state=0)

    global ap_positions
    ap_positions = get_ap_locations(X_train)

    # The models should not get to take in location as training data
    # predict_location(X_train, X_test, y_train, y_test)
    # distance_triangulation(X_train, X_test, y_train, y_test)
    distance_to_location(X_train, X_test, y_train, y_test)
    # distance_triangulation_obstacle(X_train, X_test, y_train, y_test)


def predict_location(X_train, X_test, y_train, y_test):
    """
    1. RSSI to location with RFR
    """
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

    pipeline = SplitPipeline([
            ('location', model)
        ],
        start=['^NU-AP\d{5}$'],
        targets=[]
    )

    handle_pipeline(pipeline, "Direct location prediction", X_train, X_test, y_train, y_test, search_grid=search_grid)
    
def distance_triangulation(X_train, X_test, y_train, y_test):
    """
    1. RSSI to distance with RFR
    2. Distance to location with triangulation
    """
    distance_model = RandomForestRegressor(
        n_estimators=1500, #1500
        max_depth=None,
        min_samples_split=2,
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
            ('location', TriangulationTransformer())
        ],
        start=['^NU-AP\d{5}$'],
        targets=[['^NU-AP\d{5}_distance$']], 
    )

    handle_pipeline(pipeline, "Distance-to-triangulation", X_train, X_test, y_train, y_test, search_grid=search_grid)

def distance_to_location(X_train, X_test, y_train, y_test):
    """
    1. RSSI to distance with RFR
    2. Distance to location with RFR
    """
    distance_model = RandomForestRegressor(
        n_estimators=1, #1500
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    location_model = RandomForestRegressor(
        n_estimators=1, #500
        max_depth=None,
        min_samples_split=2,
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
        ],
        start=['^NU-AP\d{5}$'],
        targets=[['^NU-AP\d{5}_distance$'], []], 
        type='cumulative'
    )


    handle_pipeline(pipeline, "Distance-to-location", X_train, X_test, y_train, y_test, search_grid=search_grid)

def distance_triangulation_obstacle(X_train, X_test, y_train, y_test):
    """
    1. RSSI to distance with RFR
    2. Distance to location with triangulation
    3. Collect obstacle data
    4. Refine location
    """
    distance_model = RandomForestRegressor(
        n_estimators=1500, #1500
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    location_model = RandomForestRegressor(
        n_estimators=1500, #175 1870
        max_depth=120, #120 None
        min_samples_split=2, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    search_grid = {
        'distance__model__n_estimators': [500, 1500, 2000],
        'distance__model__min_samples_split': [2],
        'distance__model__min_samples_leaf': [1],
        'location__n_estimators': [100, 500, 1500],
        'location__min_samples_split': [2],
        'location__min_samples_leaf': [1],
    }

    pipeline = SplitPipeline([
            ('distance', PipeLineModel(distance_model)),
            ('triangulation', TriangulationTransformer()),
            ('get_obstacles', ObstacleTransfomer()),
            ('location', location_model)
        ],
        start=['^NU-AP\d{5}$'],
        targets=[['^NU-AP\d{5}_distance$'], [], []], 
    )

    handle_pipeline(pipeline, "Distance-to-triangulation-to-obstacle", X_train, X_test, y_train, y_test, search_grid=search_grid)





def handle_pipeline(pipeline, name, X_train, X_test, y_train, y_test, search_grid):

    print(f"\n\n##### {name} ####")

    if HYPERPARAM_SEARCH:
        jobs = 1 if SOCKET else 3
        grid_search = GridSearchCV(pipeline, search_grid, n_jobs=jobs, verbose=3, error_score='raise')
        grid_search.fit(X_train, y_train)

        # Output the best parameters and score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print("Test set score: ", test_score)
    else:
        fitted_model = pipeline.fit(X_train, y_train)

        # predictions = fitted_model.predict(X_test)

        # evaluate_model(y_test, predictions, name)

        fitted_model.evaluate(X_test, y_test)

        if(SAVE_MODEL):
            now = datetime.now().strftime('%m-%d--%H-%M-%S')

            pickle.dump(fitted_model, open(f"sklearn_models/multilayer/{name}-{now}.pkl", 'wb'))
        

        



if __name__ == '__main__':
    main()