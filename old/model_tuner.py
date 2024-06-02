import math
import pandas as pd
from model import preprocess, split_data, load_files

from sklearn.metrics import classification_report, make_scorer 
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

from model import X_MAX, Y_MAX, Z_MAX

def main():
    df =  load_files(['samples F5.csv', 'samples F6.csv'])
    
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=0)

    model = RandomForestRegressor()
    mor = MultiOutputRegressor(model)
    random_search = GridSearchCV(mor, param_grid, error_score='raise', verbose=3, n_jobs=-1, scoring=score)
    # random_search = RandomizedSearchCV(mor, param_grid, error_score='raise', verbose=3, n_jobs=-1, scoring=score, n_iter=10)

    print("Start grid search")
    random_search.fit(X_train, y_train)
    print("Best estimator:", random_search.best_estimator_)
    print("Best params:", random_search.best_params_)
    print("Best score:", random_search.best_score_)
    print("Best index:", random_search.best_index_)

def prediction_loss(actual, predicted):
    worst = 0
    average = 0

    for (index, item) in enumerate(predicted):
        current = actual.iloc[index]
        current_scaled = [current.x * X_MAX, current.y * Y_MAX, current.z * Z_MAX]
        item_scaled = [item[0] * X_MAX, item[1] * Y_MAX, item[2] * Z_MAX]

        distance = math.dist(current_scaled, item_scaled)
        average += distance
        if distance > worst:
            worst = distance
    
    average /= len(predicted)

    return average

score = make_scorer(prediction_loss, greater_is_better=False)

# param_grid = { 
#     'estimator__n_estimators': [25, 50, 100, 150], 
#     'estimator__max_features': ['sqrt', 'log2', None], 
#     'estimator__max_depth': [3, 6, 9], 
#     'estimator__max_leaf_nodes': [3, 6, 9], 
# } 

param_grid = {
    'estimator__n_estimators': [1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900],
    'estimator__max_depth': [None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1],
    'estimator__max_features': ['sqrt'],
    'estimator__bootstrap': [False]
}

def exp_error(y_true, y_pred):
    # Calculate the absolute differences
    errors = np.abs(y_true - y_pred)
    # Apply an exponential function to the errors
    exp_errors = np.exp(errors)
    # Return the mean of these exponential errors
    return np.mean(exp_errors)

negative_exp_scorer = make_scorer(exp_error, greater_is_better=False)


if __name__ == "__main__":
    main()