from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from model import X_MAX, Y_MAX, Z_MAX, load_files, filter_columns, print_results, remove_columns, analyze_predictions
import sys
import pickle
from datetime import datetime
import winsound
from scipy.optimize import least_squares
import joblib


SAVE_MODEL = '--save' in sys.argv
TEMP = '--temp' in sys.argv
TUNE = '--tune' in sys.argv

def main():
    df = load_files(['samplesF5-multilayer.csv', 'samplesF6-multilayer.csv'])

    X_train, X_test, y_train, y_test, loc_train, loc_test = split_data(df)

    model = RandomForestRegressor(
        n_estimators=1000, #175 1870
        max_depth=None, #120 None
        min_samples_split=2, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
        random_state=42
    )

    print("Fitting...")
    model.fit(X_train, y_train)
    print("Done fitting")
    winsound.Beep(500, 200)
    winsound.Beep(500, 200)
    winsound.Beep(500, 200)

    # Predict on the test set
    predicted_distances = model.predict(X_test)
    predicted_locations = []

    # Trilateration to get location
    for i, distances in enumerate(predicted_distances):
        pred_x = 0
        pred_y = 0
        pred_z = 0

        totalWeight = 0

        locations = []

        for index, distance in enumerate(distances):
            ap_name = y_test.columns[index].split("_")[0]
            x = X_test[ap_name+"_x"].iloc[0]
            y = X_test[ap_name+"_y"].iloc[0]
            z = X_test[ap_name+"_z"].iloc[0]

            locations.append([x,y,z])

            weight = 1 / distance ** 2

            pred_x += x * weight
            pred_y += y * weight
            pred_z += z * weight

            totalWeight += weight
        
        trilateration = [pred_x / totalWeight, pred_y / totalWeight, pred_z / totalWeight]
        predicted_locations.append(trilateration)

    evaluate_model(y_test, predicted_distances, "Distance predictor")
    evaluate_model(loc_test, predicted_locations, "Location predictor")
    analyze_predictions(loc_test, predicted_locations)
    
    if(SAVE_MODEL):
        now = datetime.now().strftime('%m-%d--%H-%M-%S')

        pickle.dump(model, open(f"sklearn_models/multilayer-distance/model-{now}.pkl", 'wb'))



def split_data(df, test_size=0.2, random_state=0):
    # df = remove_columns(df, ['x', 'y', 'z'])
    targets, features = filter_columns(df, ['^NU-AP\d{5}_distance$'], return_removed=True)

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=random_state
    )

    loc_train, X_train = filter_columns(X_train, ['x', 'y', 'z'], return_removed=True)
    loc_test, X_test = filter_columns(X_test, ['x', 'y', 'z'], return_removed=True)

    X_train.sort_index(axis=1, inplace=True)
    X_test.sort_index(axis=1, inplace=True)

    y_train.sort_index(axis=1, inplace=True)
    y_test.sort_index(axis=1, inplace=True)

    return X_train, X_test, y_train, y_test, loc_train, loc_test

def evaluate_model(y_test, y_pred, name):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    print_results((r2, mse, mae, mdae), name)
    print("--------------------------------------------------")

def tuner():
    df = load_files(['samplesF5-multilayer.csv', 'samplesF6-multilayer.csv'])

    X_train, X_test, y_train, y_test, loc_train, loc_test = split_data(df)

    model = RandomForestRegressor()

    param_grid = {
        'n_estimators': [1000, 1200, 1400, 1600, 1800, 2000],
        'max_depth': [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'bootstrap': [False]
    }

    random_search = GridSearchCV(model, param_grid, error_score='raise', verbose=3, n_jobs=4, pre_dispatch=4)

    print("Start grid search")
    random_search.fit(X_train, y_train)
    print("Best estimator:", random_search.best_estimator_)
    print("Best params:", random_search.best_params_)
    print("Best score:", random_search.best_score_)
    print("Best index:", random_search.best_index_)



def temp():
    df = load_files(['samplesF5-multilayer.csv', 'samplesF6-multilayer.csv'])
    X_train, X_test, y_train, y_test, loc_train, loc_test = split_data(df)

    print(X_train.shape)



    # X_test.to_csv(f"data/multilayer-test-features.csv", index=False)
    # y_test.to_csv(f"data/multilayer-test-targets.csv", index=False)
    








if __name__ == "__main__":
    if TEMP:
        temp()
    elif TUNE:
        tuner()
    else:
        main()