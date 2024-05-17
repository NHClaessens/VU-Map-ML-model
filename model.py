from datetime import datetime
import math
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sys
import os
import pickle

import matplotlib.pyplot as plt


RSSI_MIN = -100
RSSI_MAX = 0
NO_SIGNAL = 1

VALIDATION_SPLIT = 0.2

REFERENCE_FLAG = "--reference" in sys.argv
EVAL_ON_EXTRAS_FLAG = "--eval-on-extras" in sys.argv
EVAL_ON_EXTRAS_FLAG = "--add-ap-count" in sys.argv

X_MAX = 70
Y_MAX = 12 if REFERENCE_FLAG else 25.9
Z_MAX = 70

PATH = "data/"

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=175, #175 1870
        max_depth=120, #120 None
        min_samples_split=4, #4 2
        min_samples_leaf=1,
        bootstrap=False,
        max_features="sqrt",
    ),
    "Decision Tree": DecisionTreeRegressor(
        criterion="poisson", 
        min_samples_split=5, 
        max_leaf_nodes=230
    ),
    "Support Vector Machine": SVR(
        degree=1, 
        coef0=1e-06, 
        gamma=1, 
        tol=1e-05, 
        C=1
    ),
    "K-Nearest Neighbours": KNeighborsRegressor(
        algorithm="kd_tree",
        leaf_size=10,
        n_neighbors=7,
        weights="distance",
        metric="manhattan",
        p=1,
    )
}

enrichment = "-ENRICHED"

def main():
    df = load_files(["samples F5.csv", "samples F6.csv"], enrichment)

    # df = filter_columns(df, ['^NU-AP\d{5}$', '^NU-AP\d{5}_obstacle_thickness$'])

    df = add_new_column(df, '^NU-AP\d{5}$', 'APs_in_range', lambda x: x != 0, [lambda x: x/324])

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.5, random_state=0)

    trained_models = {
        "Random Forest": None,
        "Decision Tree": None,
        "Support Vector Machine": None,
        "K-Nearest Neighbours": None
    }

    for (name, model) in models.items():
        model = train(model, X_train, y_train)
        predicted_values = predict(model, X_test)
        score = evaluate_model(y_test, predicted_values, name)

        trained_models[name] = (name, model, score)

        if(EVAL_ON_EXTRAS_FLAG):
            X_extra, y_extra = load_extras(['extras F5.csv', 'extras F6.csv'], "-APCount")
            X_extra = X_extra[X_train.columns]
            predicted_extras = predict(model, X_extra)
            evaluate_model(y_extra, predicted_extras, name+" On Unseen Data")

    return trained_models

def preprocess(df):
    if REFERENCE_FLAG: df = df_swap_columns(df, "y", "z")
    df = df.rename(columns={"Location X" : "x", "Location Y" : "y", "Location Z" : "z"})

    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(inplace=True)

    df = scale_rssi(df)
    df = scale_xyz(df)

    return df

def apply_pca(X_train, X_test):
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca

def df_swap_columns(df, c1, c2):
    df['temp'] = df[c1]
    
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)

    return df

def scale_rssi(df):
    for column in df.columns:
        if re.match(r'^NU-AP\d{5}$', column):
            df[column] = df[column].apply(
                lambda x: (x - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
                if x != NO_SIGNAL
                else 0
            )

    return df

def scale_xyz(df):
    for column in df.columns:
        if "x" == column or "_x" in column:
            df[column] = df[column].apply(lambda x: x / X_MAX)
        elif "y" == column or "_y" in column:
            df[column] = df[column].apply(lambda x: x / Y_MAX)
        elif "z" == column or "_z" in column:
            df[column] = df[column].apply(lambda x: x / Z_MAX)
    return df

def train(regressor, X_train, y_train):
    MOR = MultiOutputRegressor(regressor)
    # X_test = filter_columns(X_test, ['^NU-AP\d{5}$'])
    return MOR.fit(X_train, y_train)

def predict(model, X_test):
    return model.predict(X_test)

def filter_columns(df, regex_patterns, return_removed = False, fixed_columns = ['x', 'y', 'z']):
    """
    Filter DataFrame columns to keep only those specified by exact matches or regex patterns.

    Args:
    df (pd.DataFrame): The original DataFrame.
    fixed_columns (list of str): List of column names to retain.
    regex_patterns (list of str): List of regex patterns for column names to retain.

    Returns:
    pd.DataFrame: A DataFrame with only the desired columns.
    """
    # Initialize a list to hold the names of columns to keep
    columns_to_keep = fixed_columns[:]
    columns_to_remove = []
    
    # Check each column in the DataFrame against the regex patterns
    for col in df.columns:
        if any(re.match(pattern, col) for pattern in regex_patterns):
            columns_to_keep.append(col)
        else:
            columns_to_remove.append(col)
    
    # Remove duplicates from columns_to_keep
    columns_to_keep = list(set(columns_to_keep))
    columns_to_remove = list(set(columns_to_remove))
    
    # Use .filter() to keep only the specified columns
    if return_removed:
        return (df.filter(items=columns_to_keep), df.filter(items=columns_to_remove))
    else:
        return df.filter(items=columns_to_keep)


def evaluate_model(y_test, y_pred, name):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    print_results((r2, mse, mae, mdae), name)
    average = analyze_predictions(y_test, y_pred)
    print("--------------------------------------------------")
    return average


def print_results(eval_tuple, model_name):
    print(model_name)
    print("R2: ", eval_tuple[0])
    print("MSE: ", eval_tuple[1])
    print("MAE: ", eval_tuple[2])
    print("MDAE: ", eval_tuple[3])

def analyze_predictions(actual, predicted):
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
    
    print("Worst prediction error:", worst)
    print("Average prediction error:", average)
    return average


def split_data(df, target=["x", "y", "z"], test_size=0.2, random_state=0):
    features = df.loc[:, ~df.columns.isin(target)]
    targets = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def remove_columns(df, regex = []):
    for r in regex:
        df = df.loc[:, ~df.columns.str.match(r)]
    
    return df

def plot_3d(y_test, y_pred, name):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    SWAPPED = True
    if SWAPPED:
        color_map = plt.get_cmap("tab20c")
        colors = y_test["x"] + y_test["y"] + y_test["z"]
        ax.scatter3D(
            y_pred[:, 0],
            y_pred[:, 2],
            y_pred[:, 1],
            s=5,
            cmap=color_map,
            c=colors,
            label="y_pred",
        )
        ax.scatter3D(
            y_test["x"],
            y_test["z"],
            y_test["y"],
            s=20,
            cmap=color_map,
            c=colors,
            label="y_test",
        )
        ax.scatter3D(
            y_test["x"], y_test["z"], y_test["y"], s=3, color="black", label="y_test"
        )
    else:
        ax.scatter3D(
            y_test["x"], y_test["y"], y_test["z"], color="blue", label="y_test"
        )
        ax.scatter3D(
            y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], color="red", label="y_pred"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(f"Actual values vs Predicted values - {name}")
    plt.savefig(f"{name}.png", bbox_inches="tight")
    plt.show()

def export_predictions(y_test, predicted_values, addon = ""):
    predicted_df = pd.DataFrame(predicted_values, columns=['predicted_x', 'predicted_y', 'predicted_z'])

    df = pd.concat([y_test.reset_index(drop=True), predicted_df.reset_index(drop=True)], axis=1)

    for column in df.columns:
        if "x" == column or "_x" in column:
            df[column] = df[column].apply(lambda x: x * X_MAX)
        elif "y" == column or "_y" in column:
            df[column] = df[column].apply(lambda x: x * Y_MAX)
        elif "z" == column or "_z" in column:
            df[column] = df[column].apply(lambda x: x * Z_MAX)


    
    now = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    df.to_csv(f"results/{now}-{addon}.csv", index=False)


def load_files(files, options = ""):
    df_list = [pd.read_csv(PATH+file.replace(".", options+".")) for file in files]
    df = pd.concat(df_list)
    df = preprocess(df)
    return df

def load_extras(files, options = ""):
    df = load_files(files, options)

    targets = ['x', 'y', 'z']
    features = df.columns.difference(targets)

    extra_features = df[features]
    extra_targets = df[targets]

    return extra_features, extra_targets

def add_new_column(df, regex_pattern, new_column_name, predicate, processing = []):
    """
    Adds a new column to the DataFrame which counts how many columns match a given regex pattern and a predicate.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        regex_pattern (str): The regex pattern to match column names.
        new_column_name (str): The name of the new column that will store the count.
        predicate (callable): A function that takes a pd.Series and returns a Boolean Series.
        
    Returns:
        pd.DataFrame: The DataFrame with the added column.
    """
    # Compile the regex pattern for efficiency
    pattern = re.compile(regex_pattern)
    
    # Filter the DataFrame's columns based on the regex pattern
    matching_columns = [col for col in df.columns if pattern.match(col)]
    
    # Apply the predicate to each of the matching columns and sum the True values across these columns for each row
    result_series = df[matching_columns].apply(predicate).sum(axis=1)

    for process in processing:
        result_series = result_series.apply(process)
    
    # Concatenate the result as a new column
    df = pd.concat([df, result_series.rename(new_column_name)], axis=1)
    
    return df






if __name__ == "__main__":
    index = sys.argv.index('--repeat') if '--repeat' in sys.argv else -1
    repetitions = 1

    if index != -1:
        repetitions = int(sys.argv[index + 1])

    scores = {}
    
    for i in range(0, repetitions):
        scored = main()

        for item in scored.items():
            (key, value) = item

            if value == None: continue

            (name, model, score) = value

            if(not key in scores or score < scores[key][2]):
                scores[item[0]] = value

    now = datetime.now().strftime('%m-%d--%H-%M-%S')

    for item in scores.items():
        (key, value) = item
        (name, model, score) = value

        print(f"{name} got {score} as best result")

        pickle.dump(model, open(f"sklearn_models/{name}-{score}-{enrichment}-{now}.pkl", "wb"))