import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data

# model 1
# model_num = 1
# file_path = 'results/No extra data/Direct location prediction-06-21--11-03-16.csv'

# model 2
# model_num = 2
# file_path = 'results/No extra data/Distance-to-trilateration-06-21--14-35-31.csv'

# model 3
# model_num = 3
# file_path = 'results/No extra data/Distance-to-location-06-21--11-10-53.csv'

# model 4
# model_num = 4
# file_path = 'results/No extra data/Distance-to-trilateration-to-obstacle-06-21--14-51-59.csv'

predictions = {
    1: 'results/No extra data/Direct location prediction-06-21--11-03-16.csv',
    2: 'results/No extra data/Distance-to-trilateration-06-21--14-35-31.csv',
    3: 'results/No extra data/Distance-to-location-06-21--11-10-53.csv',
    4: 'results/No extra data/Distance-to-trilateration-to-obstacle-06-21--14-51-59.csv'
}


def main():
    # for key, val in predictions.items():
    #     data = pd.read_csv(val)

    #     # error_plots(data, key)
    #     rssi_variance(data, key)


    rssi_variance()


def error_plots(data, model_num):
    LOG = False
    LABELS = False
    AXIS = True
    titleSize = 30
    legendSize = 35
    axisSize = 30

    # Calculate errors for each coordinate
    data['error_x'] = data['x'] - data['pred_x']
    data['error_y'] = data['y'] - data['pred_y']
    data['error_z'] = data['z'] - data['pred_z']

    # Calculate combined Euclidean error
    data['error_combined'] = np.sqrt(data['error_x']**2 + data['error_y']**2 + data['error_z']**2)

    # Plot the error distributions

    plt.figure(figsize=(9,12))
    plt.subplots_adjust(left=0.18, bottom=0.1)
    # Plot error distribution for x
    plt.hist(data['error_x'], bins=30, alpha=0.75, color='red', edgecolor='black')
    if LOG: plt.yscale('log')
    if LABELS:
        plt.title('Error Distribution for X', fontsize=titleSize)
    if AXIS:
        plt.xlabel('Error', fontsize=legendSize)
        plt.ylabel('Frequency', fontsize=legendSize)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.savefig(f'results/No extra data/Model {model_num} error X.png', dpi=300)


    # Plot error distribution for y
    plt.figure(figsize=(9,12))
    plt.subplots_adjust(left=0.18, bottom=0.1)
    plt.hist(data['error_y'], bins=30, alpha=0.75, color='green', edgecolor='black')
    if LOG: plt.yscale('log')
    if LABELS:
        plt.title('Error Distribution for Y', fontsize=titleSize)
    if AXIS:
        plt.xlabel('Error', fontsize=legendSize)
        plt.ylabel('Frequency', fontsize=legendSize)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.savefig(f'results/No extra data/Model {model_num} error Y.png', dpi=300)

    # Plot error distribution for z
    plt.figure(figsize=(9,12))
    plt.subplots_adjust(left=0.18, bottom=0.1)
    plt.hist(data['error_z'], bins=30, alpha=0.75, color='blue', edgecolor='black')
    if LOG: plt.yscale('log')
    if LABELS:
        plt.title('Error Distribution for Z', fontsize=titleSize)
    if AXIS:
        plt.xlabel('Error', fontsize=legendSize)
        plt.ylabel('Frequency', fontsize=legendSize)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.savefig(f'results/No extra data/Model {model_num} error Z.png', dpi=300)

    # Plot combined error distribution
    plt.figure(figsize=(9,12))
    plt.subplots_adjust(left=0.18, bottom=0.1)
    plt.hist(data['error_combined'], bins=30, alpha=0.75, color='purple', edgecolor='black')
    if LOG: plt.yscale('log')
    if LABELS:
        plt.title('Combined Error Distribution', fontsize=titleSize)
    if AXIS:
        plt.xlabel('Error', fontsize=legendSize)
        plt.ylabel('Frequency', fontsize=legendSize)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.savefig(f'results/No extra data/Model {model_num} error XYZ.png', dpi=300)

    # Adjust layout
    plt.tight_layout()


    # Show the plots
    # plt.show()

def rssi_variance():

    f5 = pd.read_csv('data/samplesF5-multilayer.csv')
    f6 = pd.read_csv('data/samplesF6-multilayer.csv')

    data = pd.concat([f5, f6])

    rssi_columns = [col for col in data.columns if col.startswith('NU-AP') and not col.endswith(('_distance', '_x', '_y', '_z'))]

    # Calculate the mean RSSI for each AP at each location
    location_means = data.groupby(['x', 'y', 'z'])[rssi_columns].mean()

    # Calculate the variance of the RSSI for each AP across all locations
    ap_variances = location_means.var()

    # Plot the variance of each AP as a histogram
    plt.figure(figsize=(20, 10))
    plt.hist(ap_variances, bins=30, color='blue', edgecolor='black', alpha=0.75)
    plt.title('Histogram of RSSI Variance for Each AP Across Locations', fontsize=16)
    plt.xlabel('Variance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot as a high-resolution image
    plt.tight_layout()
    plt.savefig('RSSI_variance.png', dpi=300)




if __name__ == '__main__':
    main()
