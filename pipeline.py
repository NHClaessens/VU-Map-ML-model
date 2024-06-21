from datetime import datetime
import json
import pickle
import time
import winsound

from sklearn.discriminant_analysis import StandardScaler
from util import X_MAX, Y_MAX, Z_MAX, get_ap_locations_names, least_squares_trilaterate, load_files, filter_columns, evaluate_model, split_data, split_data_parts, trilaterate, unscale_xyz
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, ParameterGrid
import sys
import numpy as np
import pandas as pd

HYPERPARAM_SEARCH = '--hyper-search' in sys.argv
SAVE_MODEL = '--save' in sys.argv
EXPORT = '--export' in sys.argv
SOCKET = '--socket' in sys.argv
TIME = '--time' in sys.argv

NON_PROD = False
NAME_ADDITION = ''

unity = None
timer = None

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
ap_names = None

class PipeLineModel(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X, y):
        predictions =  self.model.predict(X)
        return pd.DataFrame(predictions, columns=y.columns)
    
class SplitPipeline(Pipeline):
    def __init__(self, steps, inputs, targets, remove = [], type='normal', **kwargs):
        super().__init__(steps, **kwargs)

        self.targets = targets
        self.type = type
        self.inputs = inputs
        self.remove = remove

    def fit(self, X, y):
        targets = split_data_parts(X, self.targets)
        _, X = filter_columns(X, self.remove, return_removed=True)

        for (index, step) in enumerate(self.steps[:-1]):
            name, model = step

            input = split_data_parts(X, [self.inputs[index]])[0] if self.inputs[index] else X

            model.fit(input, targets[index])

            if index < len(self.steps) - 1:
                if self.type == 'cumulative':
                    transformed = model.transform(input, targets[index])
                    transformed.index = X.index
                    X = pd.concat([X, transformed], axis=1)
                if self.type == 'normal':
                    X = model.transform(input, targets[index])

        input = split_data_parts(X, [self.inputs[-1]])[0] if self.inputs[-1] else X

        self.steps[-1][1].fit(input, y)
        
        return self
    
    def predict(self, X):
        targets = split_data_parts(X, self.targets)
        _, X = filter_columns(X, self.remove, return_removed=True)

        for (index, step) in enumerate(self.steps):
            name, model = step

            input = split_data_parts(X, [self.inputs[index]])[0] if self.inputs[index] else X

            if index < len(self.steps) - 1:
                if self.type == 'cumulative':
                    transformed = model.transform(input, targets[index])
                    transformed.index = X.index
                    X = pd.concat([X, transformed], axis=1)
                if self.type == 'normal':
                    X = model.transform(input, targets[index])
            else:
                return model.predict(input)
            
    def score(self, X, y):

        pred = self.predict(X)
        if(type(pred) is pd.DataFrame):
            pred = pred.to_numpy()
        if(type(y) is pd.DataFrame):
            y = y.to_numpy()

        return -np.mean((y - pred) ** 2)
    
    def evaluate(self, X, y):
        targets = split_data_parts(X, self.targets)
        _, X = filter_columns(X, self.remove, return_removed=True)


        for (index, step) in enumerate(self.steps):
            name, model = step

            input = split_data_parts(X, [self.inputs[index]])[0] if self.inputs[index] else X

            if index < len(self.steps) - 1:
                predictions = model.transform(input, targets[index])

                if(isinstance(model, PipeLineModel)):
                    evaluate_model(targets[index], predictions, f'Layer {index + 1}: {name}')

                if self.type == 'cumulative':
                    transformed = model.transform(input, targets[index])
                    transformed.index = X.index
                    X = pd.concat([X, transformed], axis=1)
                if self.type == 'normal':
                    X = predictions
            else:
                predictions = model.predict(input)
                evaluate_model(y, predictions, f'Layer {index + 1}: {name}', location=True)

                return predictions

class TrilaterationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_closest_points=324) -> None:
        super().__init__()
        self.n_closest_points = n_closest_points

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        res = X.apply(self.trilaterate, axis=1)
        return res
    
    def predict(self, X):
        return self.transform(X)
    
    def trilaterate(self, distances):
        # loc =  trilaterate(distances, ap_positions)
        loc =  least_squares_trilaterate(distances, ap_positions, self.n_closest_points)
        return pd.Series({
            'x': loc[0],
            'y': loc[1],
            'z': loc[2],
        })

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
    
    def transform(self, X, y):
        res =  X.apply(self.get_obstacles, axis=1)
        return res


    def get_obstacles(self, pos):
        unity.sendall(json.dumps({
            "type": "obstacles",
            # "ap_names": ap_names,
            "data": {
                "x": pos.x * X_MAX,
                "y": pos.y * Y_MAX,
                "z": pos.z * Z_MAX,
            }
        }).encode())
        
        data = unity.recv(10000).decode()

        data = json.loads(data)
        obstacles = json.loads(data['data']['obstacle_thickness'])

        d = {}

        for (index, dist) in enumerate(obstacles):
            d[ap_names[index]+'_obstacle_thickness'] = dist

        return pd.Series(d)

def main():
    df = load_files(["samplesF5-multilayer.csv", "samplesF6-multilayer.csv"])
    # df = load_files(["samples F5-ENRICHED.csv", "samples F6-ENRICHED.csv"])
    # df = load_files(["samples F5 everything.csv", "samples F6 everything.csv"])

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

    # X_train = X_train[0:10]
    # X_test = X_test[0:10]
    # y_train = y_train[0:10]
    # y_test = y_test[0:10]

    global ap_positions
    global ap_names
    ap_positions, ap_names = get_ap_locations_names(X_train)

    # The models should not get to take in location as training data
    # predict_location(X_train, X_test, y_train, y_test)
    # distance_trilateration(X_train, X_test, y_train, y_test)
    # distance_to_location(X_train, X_test, y_train, y_test)
    distance_trilateration_obstacle(X_train, X_test, y_train, y_test)


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
        inputs=[[]] if NON_PROD else [['^NU-AP\d{5}$']],
        targets=[],
        remove=['^NU-AP\d{5}_distance$']
    )

    handle_pipeline(pipeline, "Direct location prediction", X_train, X_test, y_train, y_test, search_grid=search_grid)
    
def distance_trilateration(X_train, X_test, y_train, y_test):
    """
    1. RSSI to distance with RFR
    2. Distance to location with trilateration
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
        # 'distance__model__n_estimators': [100, 1500, 2000],
        # 'distance__model__min_samples_split': [2],
        # 'distance__model__min_samples_leaf': [1],
        'location__n_closest_points': [50, 100, 150, 200, 250, 300, 324],
    }

    pipeline = SplitPipeline([
            ('distance', PipeLineModel(distance_model)),
            ('location', TrilaterationTransformer(n_closest_points=324))
        ],
        inputs=[[], []] if NON_PROD else [['^NU-AP\d{5}$'], []],
        targets=[['^NU-AP\d{5}_distance$']], 
        remove=['^NU-AP\d{5}_distance$']
    )

    handle_pipeline(pipeline, "Distance-to-trilateration", X_train, X_test, y_train, y_test, search_grid=search_grid)

def distance_to_location(X_train, X_test, y_train, y_test):
    """
    1. RSSI to distance with RFR
    2. Distance to location with RFR
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
        n_estimators=1500, #500
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
        inputs=[[], []] if NON_PROD else [['^NU-AP\d{5}$'], []],
        targets=[['^NU-AP\d{5}_distance$'], []], 
        type='cumulative',
        remove=['^NU-AP\d{5}_distance$']
    )


    handle_pipeline(pipeline, "Distance-to-location", X_train, X_test, y_train, y_test, search_grid=search_grid)

def distance_trilateration_obstacle(X_train, X_test, y_train, y_test):
    """
    1. RSSI to distance with RFR
    2. Distance to location with trilateration
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
            ('trilateration', TrilaterationTransformer()),
            ('get_obstacles', ObstacleTransfomer()),
            ('location', location_model)
        ],
        inputs=[['^NU-AP\d{5}$'], ['^NU-AP\d{5}_distance$'], ['^(x|y|z)$'], []],
        targets=[['^NU-AP\d{5}_distance$'], [], []], 
        type='cumulative',
        remove=['^NU-AP\d{5}_distance$']
    )

    handle_pipeline(pipeline, "Distance-to-trilateration-to-obstacle", X_train, X_test, y_train, y_test, search_grid=search_grid) 




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
        start = time.time()

        fitted_model = pipeline.fit(X_train, y_train)

        print(f"{name} took {time.time() - start} to fit on {len(X_train)} samples")

        fitted_model.evaluate(X_test, y_test)

        start = time.time()
        predictions = fitted_model.predict(X_test)
        # print("predictions", predictions)
        # print("actual", y_test)
        print(f"{name} took {time.time() - start} to predict on {len(X_test)} samples")


        if(EXPORT):
            now = datetime.now().strftime('%m-%d--%H-%M-%S')

            if type(predictions) is pd.DataFrame:
                predictions.rename(columns={
                    'x': 'pred_x',
                    'y': 'pred_y',
                    'z': 'pred_z',
                }, inplace=True)
                predictions.index = y_test.index
            else:
                predictions = pd.DataFrame(predictions, columns=['pred_x', 'pred_y', 'pred_z'], index=y_test.index)

            export = pd.concat([y_test, predictions], axis=1)
            export = unscale_xyz(export)

            export.to_csv(f"results/{name}{NAME_ADDITION}-{now}{'-non-prod' if NON_PROD else ''}.csv", index=False)

        if(SAVE_MODEL):
            now = datetime.now().strftime('%m-%d--%H-%M-%S')

            pickle.dump(fitted_model, open(f"sklearn_models/multilayer/{name}{NAME_ADDITION}-{now}.pkl", 'wb'))
        

        



if __name__ == '__main__':
    main()
    winsound.Beep(500, 200)
    winsound.Beep(500, 200)
    winsound.Beep(500, 200)