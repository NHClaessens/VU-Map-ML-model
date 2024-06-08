import re
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import math
from datetime import datetime


RSSI_MIN = -100
RSSI_MAX = 0
NO_SIGNAL = 1

VALIDATION_SPLIT = 0.2

REFERENCE_FLAG = "--reference" in sys.argv
EVAL_ON_EXTRAS_FLAG = "--eval-on-extras" in sys.argv
EVAL_ON_EXTRAS_FLAG = "--add-ap-count" in sys.argv

X_MAX = 70
Y_MAX = 12 if REFERENCE_FLAG else 49.9
Z_MAX = 70

PATH = "data/"


def preprocess(df):
    df = df.rename(columns={"Location X" : "x", "Location Y" : "y", "Location Z" : "z"})

    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(inplace=True)

    df = scale_rssi(df)
    df = scale_xyz(df)

    return df

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

def filter_columns(df, regex_patterns, return_removed = False):
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
    columns_to_keep = []
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

    keep = df.filter(items=columns_to_keep)
    keep.sort_index(axis=1, inplace=True)

    # Use .filter() to keep only the specified columns
    if return_removed:
        remove = df.filter(items=columns_to_remove)
        remove.sort_index(axis=1, inplace=True)
        return keep, remove
    else:
        return keep


def evaluate_model(y_test, y_pred, name, location=False):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    print_results((r2, mse, mae, mdae), name)

    if(location):
        analyze_predictions(y_test, y_pred)
    print("--------------------------------------------------")


def print_results(eval_tuple, model_name):
    print(model_name)
    print("R2: ", eval_tuple[0])
    print("MSE: ", eval_tuple[1])
    print("MAE: ", eval_tuple[2])
    print("MDAE: ", eval_tuple[3])

def analyze_predictions(actual, predicted):
    worst = 0
    average = 0

    if isinstance(predicted, pd.DataFrame):
        for index, item in predicted.iterrows():
            current = actual.iloc[index]
            current_scaled = [current.x * X_MAX, current.y * Y_MAX, current.z * Z_MAX]
            item_scaled = [item.x * X_MAX, item.y * Y_MAX, item.z * Z_MAX]

            distance = math.dist(current_scaled, item_scaled)
            average += distance
            if distance > worst:
                worst = distance
    else:
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

    X_train.sort_index(axis=1, inplace=True)
    X_test.sort_index(axis=1, inplace=True)
    y_train.sort_index(axis=1, inplace=True)
    y_test.sort_index(axis=1, inplace=True)

    return X_train, X_test, y_train, y_test

def remove_columns(df, regex = []):
    for r in regex:
        df = df.loc[:, ~df.columns.str.match(r)]
    
    return df

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

def split_data_loc(df, test_size=0.2, random_state=0):
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

def split_data_parts(df, regex_list=None):
    """
    Split the DataFrame into train and test sets based on targets and optionally split the features based on regex_list.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (list): The list of target column names.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    regex_list (list of lists): A list of lists of regexes to split the features into parts.

    Returns:
    dict: A dictionary with the split parts as keys and their corresponding train-test splits as values.
    """

    split_parts = []
    for i, regex_group in enumerate(regex_list):
        part = pd.DataFrame()
        for regex in regex_group:
            part = pd.concat([part, df.filter(regex=regex)], axis=1)

        part.sort_index(axis=1, inplace=True)

        split_parts.append(part)

    return split_parts

def filter_tuple(tup, remove_value=None):
    """
    Filters a tuple to remove all instances of remove_value.

    Parameters:
    tup (tuple): The original tuple.
    remove_value: The value to remove from the tuple.

    Returns:
    tuple: The filtered tuple with remove_value removed.
    """
    return tuple(x for x in tup if x is not remove_value)

def triangulate(distances, ap_positions):
    pred_x = 0
    pred_y = 0
    pred_z = 0

    totalWeight = 0

    for index, distance in enumerate(distances):
        x = ap_positions[index][0]
        y = ap_positions[index][1]
        z = ap_positions[index][2]

        weight = 1 / distance ** 2

        pred_x += x * weight
        pred_y += y * weight
        pred_z += z * weight

        totalWeight += weight
    
    trilateration = [pred_x / totalWeight, pred_y / totalWeight, pred_z / totalWeight]
    return trilateration

def get_ap_locations_names(df):
    ap_names = df.filter(regex='^NU-AP\d{5}$')


    locations = []
    for name in ap_names:
        locations.append([
            df[name+'_x'].iloc[0],
            df[name+'_y'].iloc[0],
            df[name+'_z'].iloc[0],
        ])
    
    return locations, list(ap_names.columns)