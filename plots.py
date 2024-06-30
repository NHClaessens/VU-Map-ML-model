import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata

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
titleSize = 30
legendSize = 35
axisSize = 30


def main():
    for key, val in predictions.items():
        data = pd.read_csv(val)

        # error_plots(data, key)
        # coordinate_hist(data, key)

    # rssi_vs_distance()
    # rssi_and_obstacle_vs_distance()
    rssi_and_obstacle_vs_distance_lineplot()
    rssi_variance_and_obstacle_vs_distance_lineplot()
    # rssi_variance()


def error_plots(data, model_num):
    LOG = False
    LABELS = False
    AXIS = True

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

def coordinate_hist(data, model_num):
    model1 = pd.read_csv(predictions[1])
    model2 = pd.read_csv(predictions[2])
    model3 = pd.read_csv(predictions[3])
    model4 = pd.read_csv(predictions[4])

    # Define function to save individual histogram
    def save_histogram(data, column, colors, title, output_path, bins=50):
        plt.figure(figsize=(20, 10))
        plt.hist([data[column], data['pred_'+column]], bins=bins, alpha=1, color=['red', 'cyan'], edgecolor='black', linewidth=1.2, label=[f'Real {column.upper()}'])
        
        # plt.hist([data[column]], bins=bins, alpha=0.5, color='red', edgecolor='black', linewidth=1.2, label=[f'Real {column.upper()}'])
        # plt.hist([model1['pred_'+column]], bins=bins, alpha=0.5, color='blue', edgecolor='black', linewidth=1.2, label=[f'Predicted {column.upper()} model 1'])
        # plt.hist([model2['pred_'+column]], bins=bins, alpha=0.5, color='yellow', edgecolor='black', linewidth=1.2, label=[f'Predicted {column.upper()} model 2'])
        # plt.hist([model3['pred_'+column]], bins=bins, alpha=0.5, color='green', edgecolor='black', linewidth=1.2, label=[f'Predicted {column.upper()} model 3'])
        # plt.hist([model4['pred_'+column]], bins=bins, alpha=0.5, color='purple', edgecolor='black', linewidth=1.2, label=[f'Predicted {column.upper()} model 4'])
        
        
        plt.xlabel(f'{column.upper()} values', fontsize=legendSize)
        plt.ylabel('Frequency', fontsize=legendSize)
        plt.tick_params(axis='both', which='major', labelsize=axisSize)
        plt.legend(fontsize=legendSize)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Define color schemes
    colors = {
        'x': ['red', 'cyan'], 
        'y': ['green', 'magenta'], 
        'z': ['blue', 'orange']
    }
    # colors = {
    #     'x': 'red', 
    #     'y': 'green', 
    #     'z': 'blue'
    # }

    # Save histograms for x, y, and z values
    save_histogram(data, 'x', colors['x'], f'Histogram of X values', f'results/No extra data/Model {model_num} hist_x.png', bins=70)
    save_histogram(data, 'y', colors['y'], f'Histogram of Y values', f'results/No extra data/Model {model_num} hist_y.png', bins=50)
    save_histogram(data, 'z', colors['z'], f'Histogram of Z values', f'results/No extra data/Model {model_num} hist_z.png', bins=70)
    # save_histogram(data, 'x', colors['x'], f'Histogram of X values', f'results/No extra data/hist_x.png', bins=70)
    # save_histogram(data, 'y', colors['y'], f'Histogram of Y values', f'results/No extra data/hist_y.png', bins=50)
    # save_histogram(data, 'z', colors['z'], f'Histogram of Z values', f'results/No extra data/hist_z.png', bins=70)

    # results/No extra data/Model {model_num} hist_x.png

def rssi_vs_distance():
    df1 = pd.read_csv('data/samples F5 everything.csv')
    df2 = pd.read_csv('data/samples F5 everything.csv')

    data = pd.concat([df1, df2])

    # Extract the columns with RSSI values and corresponding distance columns
    rssi_columns = [col for col in data.columns if col.startswith('NU-AP') and not col.endswith('_distance')]
    distance_columns = [col for col in data.columns if col.endswith('_distance')]

    # Initialize lists to hold all RSSI values and distances
    all_rssi_values = []
    all_distances = []

    # Iterate over each RSSI column and corresponding distance column
    for rssi_col, distance_col in zip(rssi_columns, distance_columns):
        rssi_values = data[rssi_col].dropna()
        distances = data[distance_col].dropna()
        
        # Filter out values where either RSSI or distance is 0
        valid_indices = (rssi_values != 1)
        all_rssi_values.extend(rssi_values[valid_indices].tolist())
        all_distances.extend(distances[valid_indices].tolist())

    # Create a scatter plot
    plt.figure(figsize=(15, 10))
    plt.scatter(all_distances, all_rssi_values, alpha=0.5)
    plt.xlabel('Distance', fontsize=legendSize)
    plt.ylabel('RSSI Value', fontsize=legendSize)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.grid(True)
    plt.show()

def rssi_and_obstacle_vs_distance():
    df1 = pd.read_csv('data/samples F5 everything.csv')
    df2 = pd.read_csv('data/samples F6 everything.csv')

    df = pd.concat([df1, df2])

    # Extract the columns with RSSI values, corresponding distance, and obstacle thickness columns
    rssi_columns = [col for col in df.columns if col.startswith('NU-AP') and not col.endswith(('_distance', '_thickness'))]
    distance_columns = [col for col in df.columns if col.endswith('_distance')]
    thickness_columns = [col for col in df.columns if col.endswith('_obstacle_thickness')]

    # Initialize lists to hold all RSSI values, distances, and thicknesses
    all_rssi_values = []
    all_distances = []
    all_thicknesses = []

    # Iterate over each RSSI column and corresponding distance and thickness columns
    for rssi_col, distance_col, thickness_col in zip(rssi_columns, distance_columns, thickness_columns):
        rssi_values = df[rssi_col].dropna()
        distances = df[distance_col].dropna()
        thicknesses = df[thickness_col].dropna()
        
        # Filter out values where either RSSI, distance, or thickness is 0
        valid_indices = (rssi_values != 1)
        all_rssi_values.extend(rssi_values[valid_indices].tolist())
        all_distances.extend(distances[valid_indices].tolist())
        all_thicknesses.extend(thicknesses[valid_indices].tolist())

    # Create a DataFrame for the aggregated data
    data = pd.DataFrame({
        'RSSI': all_rssi_values,
        'Distance': all_distances,
        'Thickness': all_thicknesses
    })

    distance_binsize = 2
    thickness_binsize = 0.5

    # Define finer bins for distance and thickness
    distance_bins = np.arange(0, max(all_distances) + distance_binsize, distance_binsize)
    thickness_bins = np.arange(0, max(all_thicknesses) + thickness_binsize, thickness_binsize)

    # Create a pivot table to calculate the average RSSI for each distance and thickness bin
    pivot_table = data.pivot_table(values='RSSI', index=pd.cut(data['Thickness'], bins=thickness_bins),
                                   columns=pd.cut(data['Distance'], bins=distance_bins), aggfunc='mean')

    # Create a mesh grid for interpolation
    distance_grid, thickness_grid = np.meshgrid(distance_bins, thickness_bins)
    points = np.array([data['Distance'], data['Thickness']]).T
    values = data['RSSI']

    # Interpolate missing values using griddata
    # interpolated_grid = griddata(points, values, (distance_grid, thickness_grid), method='linear')

    # Plot the pivot table and interpolated grid
    plt.figure(figsize=(12, 8))

    plt.imshow(pivot_table, aspect='auto', cmap='viridis', origin='lower',
               extent=[distance_bins[0], distance_bins[-1], thickness_bins[0], thickness_bins[-1]])
    # plt.imshow(interpolated_grid, aspect='auto', alpha=0.5, cmap='viridis', origin='lower',
    #            extent=[distance_bins[0], distance_bins[-1], thickness_bins[0], thickness_bins[-1]])
    
    # plt.imshow(interpolated_grid.T, aspect='auto', cmap='viridis', origin='lower',
    #            extent=[distance_bins.min(), distance_bins.max(), thickness_bins.min(), thickness_bins.max()])
    
    plt.colorbar(label='Average RSSI')
    plt.title('RSSI Value Heatmap with Interpolated Values')
    plt.xlabel('Distance')
    plt.ylabel('Obstacle Thickness')
    plt.show()
































    # # Extract the columns with RSSI values, corresponding distance, and obstacle thickness columns
    # rssi_columns = [col for col in df.columns if col.startswith('NU-AP') and not col.endswith(('_distance', '_thickness'))]
    # distance_columns = [col for col in df.columns if col.endswith('_distance')]
    # thickness_columns = [col for col in df.columns if col.endswith('_obstacle_thickness')]

    # # Initialize lists to hold all RSSI values, distances, and thicknesses
    # all_rssi_values = []
    # all_distances = []
    # all_thicknesses = []

    # # Iterate over each RSSI column and corresponding distance and thickness columns
    # for rssi_col, distance_col, thickness_col in zip(rssi_columns, distance_columns, thickness_columns):
    #     rssi_values = df[rssi_col].dropna()
    #     distances = df[distance_col].dropna()
    #     thicknesses = df[thickness_col].dropna()
        
    #     # Filter out values where either RSSI, distance, or thickness is 0
    #     valid_indices = (rssi_values != 0) & (distances != 0) & (thicknesses != 0)
    #     all_rssi_values.extend(rssi_values[valid_indices].tolist())
    #     all_distances.extend(distances[valid_indices].tolist())
    #     all_thicknesses.extend(thicknesses[valid_indices].tolist())

    # # Create a DataFrame for the aggregated data
    # data = pd.DataFrame({
    #     'RSSI': all_rssi_values,
    #     'Distance': all_distances,
    #     'Thickness': all_thicknesses
    # })

    # # Define bins for thickness
    # thickness_bins = np.arange(0, max(all_thicknesses) + 1, 0.5)
    # data['Thickness_Bin'] = pd.cut(data['Thickness'], bins=thickness_bins, right=False)

    # # Plot RSSI vs Distance for each thickness bin with error boundaries
    # plt.figure(figsize=(14, 8))  # Increase the figure size for better legend fitting
    # colors = plt.cm.viridis(np.linspace(0, 1, len(thickness_bins)))

    # for i, (name, group) in enumerate(data.groupby('Thickness_Bin')):
    #     if i < 3:
    #         continue
    #     if not group.empty:
    #         # Calculate mean, min, and max for each distance
    #         mean_rssi = group.groupby('Distance')['RSSI'].mean()
    #         min_rssi = group.groupby('Distance')['RSSI'].min()
    #         max_rssi = group.groupby('Distance')['RSSI'].max()
            
    #         # Apply rolling mean for smoothing
    #         mean_rssi_smooth = mean_rssi.rolling(window=5, min_periods=1).mean()
    #         min_rssi_smooth = min_rssi.rolling(window=5, min_periods=1).mean()
    #         max_rssi_smooth = max_rssi.rolling(window=5, min_periods=1).mean()
            
    #         # Plot line with min-max error boundaries
    #         distances = mean_rssi.index
    #         plt.plot(distances, mean_rssi_smooth, color=colors[i], label=f'Thickness {name}')
    #         # plt.fill_between(distances, min_rssi_smooth, max_rssi_smooth, color=colors[i], alpha=0.2)

    # plt.title('RSSI Value vs. Distance Grouped by Obstacle Thickness')
    # plt.xlabel('Distance')
    # plt.ylabel('RSSI Value')
    # plt.legend(title='Obstacle Thickness (m)', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True)
    # plt.subplots_adjust(right=0.75)  # Adjust subplot parameters to make room for the legend
    # plt.show()

























    # print(f'Total items, {len(df)}')

    # # Extract the columns with RSSI values, corresponding distance, and obstacle thickness columns
    # rssi_columns = [col for col in df.columns if col.startswith('NU-AP') and not col.endswith(('_distance', '_thickness'))]
    # distance_columns = [col for col in df.columns if col.endswith('_distance')]
    # thickness_columns = [col for col in df.columns if col.endswith('_obstacle_thickness')]

    # # Initialize lists to hold all RSSI values, distances, and thicknesses
    # all_rssi_values = []
    # all_distances = []
    # all_thicknesses = []

    # # Iterate over each RSSI column and corresponding distance and thickness columns
    # for rssi_col, distance_col, thickness_col in zip(rssi_columns, distance_columns, thickness_columns):
    #     rssi_values = df[rssi_col].dropna()
    #     distances = df[distance_col].dropna()
    #     thicknesses = df[thickness_col].dropna()
        
    #     # Filter out values where either RSSI, distance, or thickness is 0
    #     valid_indices = (rssi_values != 1)
    #     all_rssi_values.extend(rssi_values[valid_indices].tolist())
    #     all_distances.extend(distances[valid_indices].tolist())
    #     all_thicknesses.extend(thicknesses[valid_indices].tolist())

    # # Create a DataFrame for the aggregated data
    # data = pd.DataFrame({
    #     'RSSI': all_rssi_values,
    #     'Distance': all_distances,
    #     'Thickness': all_thicknesses
    # })

    # # Define bins for thickness
    # thickness_bins = np.arange(0, max(all_thicknesses) + 1, 0.5)
    # data['Thickness_Bin'] = pd.cut(data['Thickness'], bins=thickness_bins, right=False)

    # # Plot RSSI vs Distance for each thickness bin
    # plt.figure(figsize=(14, 8))  # Increase the figure size for better legend fitting
    # colors = plt.cm.viridis(np.linspace(0, 1, len(thickness_bins)))

    # for i, (name, group) in enumerate(data.groupby('Thickness_Bin')):
    #     print(f'Group {name} contains {len(group)} items')
    #     if not group.empty:
    #         plt.scatter(group['Distance'], group['RSSI'], color=colors[i], alpha=0.5, label=f'Thickness {name}')

    # plt.title('RSSI Value vs. Distance Grouped by Obstacle Thickness')
    # plt.xlabel('Distance')
    # plt.ylabel('RSSI Value')
    # plt.legend(title='Obstacle Thickness (m)', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True)
    # plt.subplots_adjust(right=0.75)  # Adjust subplot parameters to make room for the legend
    # plt.show()

def rssi_and_obstacle_vs_distance_lineplot():
    df1 = pd.read_csv('data/samples F5 everything.csv')
    df2 = pd.read_csv('data/samples F6 everything.csv')

    df = pd.concat([df1, df2])

    # Extract the columns with RSSI values, corresponding distance, and obstacle thickness columns
    rssi_columns = [col for col in df.columns if col.startswith('NU-AP') and not col.endswith(('_distance', '_thickness'))]
    distance_columns = [col for col in df.columns if col.endswith('_distance')]
    thickness_columns = [col for col in df.columns if col.endswith('_obstacle_thickness')]

    # Initialize lists to hold all RSSI values, distances, and thicknesses
    all_rssi_values = []
    all_distances = []
    all_thicknesses = []

    # Iterate over each RSSI column and corresponding distance and thickness columns
    for rssi_col, distance_col, thickness_col in zip(rssi_columns, distance_columns, thickness_columns):
        rssi_values = df[rssi_col].dropna()
        distances = df[distance_col].dropna()
        thicknesses = df[thickness_col].dropna()
        
        # Filter out values where RSSI is 1
        valid_indices = (rssi_values != 1)
        all_rssi_values.extend(rssi_values[valid_indices].tolist())
        all_distances.extend(distances[valid_indices].tolist())
        all_thicknesses.extend(thicknesses[valid_indices].tolist())

    # Create a DataFrame for the aggregated data
    data = pd.DataFrame({
        'RSSI': all_rssi_values,
        'Distance': all_distances,
        'Thickness': all_thicknesses
    })

    distance_binsize = 2
    thickness_binsize = 0.5

    # Define bins for distance and thickness
    distance_bins = np.arange(0, max(all_distances) + distance_binsize, distance_binsize)
    thickness_bins = np.arange(0, max(all_thicknesses) + thickness_binsize, thickness_binsize)

    # Create a pivot table to calculate the average RSSI for each distance and thickness bin
    pivot_table = data.pivot_table(values='RSSI', index=pd.cut(data['Thickness'], bins=thickness_bins),
                                   columns=pd.cut(data['Distance'], bins=distance_bins), aggfunc='mean')
    
    # Interpolate missing values to ensure continuous lines
    pivot_table = pivot_table.apply(lambda x: x.interpolate(limit_direction='both'))

    cmap = ListedColormap(plt.cm.get_cmap('tab20').colors)
    colors = cmap(np.linspace(0, 1, len(pivot_table)))

    # Plot each thickness bin as a separate line
    plt.figure(figsize=(20, 8), dpi=300)
    # colors = plt.cm.viridis(np.linspace(0, 1, len(pivot_table)))

    for i, (thickness_bin, row) in enumerate(pivot_table.iterrows()):
        plt.plot(distance_bins[:-1], row, label=f'Thickness: {thickness_bin}', color=colors[i], alpha=0.85)

    plt.xlabel('Distance', fontsize=legendSize)
    plt.ylabel('Average RSSI', fontsize=legendSize)
    # plt.title('Average RSSI vs Distance for Different Thickness Bins')
    plt.legend(title='Obstacle Thickness (m)', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.grid(True)
    plt.subplots_adjust(left=0.08, right=0.74, bottom=0.12)
    plt.savefig(f'results/No extra data/RSSI vs distance.png', dpi=300)

def rssi_variance_and_obstacle_vs_distance_lineplot():
    df1 = pd.read_csv('data/samples F5 everything.csv')
    df2 = pd.read_csv('data/samples F6 everything.csv')

    df = pd.concat([df1, df2])

    # Extract the columns with RSSI values, corresponding distance, and obstacle thickness columns
    rssi_columns = [col for col in df.columns if col.startswith('NU-AP') and not col.endswith(('_distance', '_thickness'))]
    distance_columns = [col for col in df.columns if col.endswith('_distance')]
    thickness_columns = [col for col in df.columns if col.endswith('_obstacle_thickness')]

    # Initialize lists to hold all RSSI values, distances, and thicknesses
    all_rssi_values = []
    all_distances = []
    all_thicknesses = []

    # Iterate over each RSSI column and corresponding distance and thickness columns
    for rssi_col, distance_col, thickness_col in zip(rssi_columns, distance_columns, thickness_columns):
        rssi_values = df[rssi_col].dropna()
        distances = df[distance_col].dropna()
        thicknesses = df[thickness_col].dropna()
        
        # Filter out values where RSSI is 1
        valid_indices = (rssi_values != 1)
        all_rssi_values.extend(rssi_values[valid_indices].tolist())
        all_distances.extend(distances[valid_indices].tolist())
        all_thicknesses.extend(thicknesses[valid_indices].tolist())

    # Create a DataFrame for the aggregated data
    data = pd.DataFrame({
        'RSSI': all_rssi_values,
        'Distance': all_distances,
        'Thickness': all_thicknesses
    })

    distance_binsize = 2
    thickness_binsize = 0.5

    # Define bins for distance and thickness
    distance_bins = np.arange(0, max(all_distances) + distance_binsize, distance_binsize)
    thickness_bins = np.arange(0, max(all_thicknesses) + thickness_binsize, thickness_binsize)

    # Create a pivot table to calculate the average RSSI for each distance and thickness bin
    pivot_table = data.pivot_table(values='RSSI', index=pd.cut(data['Thickness'], bins=thickness_bins),
                                   columns=pd.cut(data['Distance'], bins=distance_bins), aggfunc= lambda x: x.max() - x.min())
    
    # Interpolate missing values to ensure continuous lines
    pivot_table = pivot_table.apply(lambda x: x.interpolate(limit_direction='both'))

    cmap = ListedColormap(plt.cm.get_cmap('tab20').colors)
    colors = cmap(np.linspace(0, 1, len(pivot_table)))

    # Plot each thickness bin as a separate line
    plt.figure(figsize=(20, 8), dpi=300)
    # colors = plt.cm.viridis(np.linspace(0, 1, len(pivot_table)))

    for i, (thickness_bin, row) in enumerate(pivot_table.iterrows()):
        plt.plot(distance_bins[:-1], row, label=f'Thickness: {thickness_bin}', color=colors[i], alpha=0.85)

    plt.xlabel('Distance', fontsize=legendSize)
    plt.ylabel('RSSI variance', fontsize=legendSize)
    # plt.title('Average RSSI vs Distance for Different Thickness Bins')
    plt.legend(title='Obstacle Thickness (m)', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=25 )
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.grid(True)
    plt.subplots_adjust(left=0.06, right=0.74, bottom=0.12)
    plt.savefig(f'results/No extra data/RSSI variance vs distance.png', dpi=300)


if __name__ == '__main__':
    main()
