#Import libraries and functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

#######################################################################################################################
scaler = StandardScaler()
def clean_string(s):
    s = s.replace('_', ' ')  # replace underscores with spaces
    s = s.title()  # capitalize the first letter of each word
    return s

#######################################################################################################################
#Behavior Downsampling Function

def downsample_neural_data(neural_data, frequency):
    """
    Downsample neural data to a specified frequency.

    Args:
    - neural_data (pd.DataFrame): A Pandas DataFrame containing the neural data.
    - frequency (str): The frequency to downsample to, in Pandas resample format (e.g., '100ms').

    Returns:
    - ds_neural_data (pd.DataFrame): A Pandas DataFrame containing the downsampled neural data.
    """
    # Ensure the 'Time' column exists and is not in the data columns
    if 'Time' not in neural_data.columns:
        raise ValueError("'Time' column is missing in the input DataFrame.")
    
    # Set 'Time' as the index after converting it to timedelta
    neural_data = neural_data.set_index(pd.to_timedelta(neural_data['Time'], unit='s')).drop(columns=['Time'])

    # Create an empty DataFrame to store downsampled data
    ds_neural_data = pd.DataFrame()

    # Retrieve a list of neuron (cell) column names
    list_of_neuron_names = list(neural_data.columns)

    # Loop through each neuron column, downsample, and store the result in a new DataFrame
    for neuron in list_of_neuron_names:
        # Remove missing values to ensure proper binning
        output = neural_data[neuron].dropna()
        
        # Downsample by taking the mean of observations within the specified time range
        output = output.resample(frequency).mean()
        
        # Store the downsampled neuron data in the ds_neural_data DataFrame
        ds_neural_data[neuron] = output
    
    # Convert the final DataFrame index to total seconds
    ds_neural_data.index = ds_neural_data.index.total_seconds()
    
    return ds_neural_data
################################################################

  # Neural Data Downsampling Function
# Here we are defining the neural data downsampling function
# Named “def downsample_neural_data(neural_data, frequency):” in functions3 library
def downsample_neural_data(neural_data, frequency):
    """
    Downsample neural data to a specified frequency.

    Args:
    - neural_data (pd.DataFrame): A Pandas DataFrame containing the neural data.
    - frequency (str): The frequency to downsample to, in Pandas resample format (e.g., '100ms').

    Returns:
    - ds_neural_data (pd.DataFrame): A Pandas DataFrame containing the downsampled neural data.
    """
    # Create an empty DataFrame where downsampled columns will be stored
    ds_neural_data = pd.DataFrame()
    
    # Remove the 'Time' column from the matrix and set it as the index (convert to timedelta in seconds)
    neural_data = neural_data.set_index(pd.to_timedelta(neural_data['Time'], unit='s'))

    # Retrieve a list of neuron (cell) column names
    list_of_neuron_names = list(neural_data.columns)

    # Loop through each neuron column, downsample, and store the result in a new DataFrame
    for neuron in list_of_neuron_names:
        # Remove missing values to ensure proper binning
        output = neural_data[neuron].dropna()
        
        # Downsample by taking the mean of observations within the specified time range
        output = output.resample(frequency).mean()
        
        # Store the downsampled neuron data in the ds_neural_data DataFrame
        ds_neural_data[neuron] = output
    
    # Set the index of the final DataFrame to total seconds
    ds_neural_data.index = ds_neural_data.index.total_seconds()
    
    return ds_neural_data

####################################################################################################
# Function to plot multicollinearity diagnostics across multiple datasets
def plot_diagnostics_analysis(df_list, title_list):
    """
    Plot multicollinearity diagnostics for a list of data frames.

    The purpose is to identify variables that may be too dependent on one another. 
    Create a list with the behavioral data frames you wish to plot for all mice and across sessions 
    since diagnostics may differ between mice or across learning.

    Args:
    - df_list (list): A list of Pandas DataFrames containing behavioral and task data.
      Example: df_list = [beh_data_M1_D1, beh_data_M2_D1, ...]
    - title_list (list): A list with the titles (str) for each data frame inside df_list.
      Example: title_list = ['Mouse 1', 'Mouse 2', ...]

    Returns:
    - Generates diagnostic plots for each data frame in df_list, including correlation heatmaps, 
      VIF values, condition numbers, and minimum eigenvalues.
    """
    
    # Set up font size for the graphs
    plt.rcParams['font.size'] = 14
    fig, axs = plt.subplots(3, 5, figsize=(55, 24), gridspec_kw={'hspace': 0.8, 'wspace': 0.5})

    for idx, df in enumerate(df_list):
        # Compute correlation matrix for each DataFrame
        corr = df.corr()

        # Create a mask to visualize only the upper triangle of the correlation matrix
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Define a color map for the heatmap plot
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Plot the correlation heatmap
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.5}, ax=axs[0, idx])
        axs[0, idx].set_title(f'Correlation Heatmap - {title_list[idx]}')

        # Variance Inflation Factor (VIF) calculation
        vif = pd.DataFrame()
        vif["variables"] = df.columns
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

        # Plot VIF values
        vif.sort_values("VIF", ascending=False).plot(x="variables", y="VIF", kind='bar', legend=False, ax=axs[1, idx])
        axs[1, idx].set_title(f'VIF values - {title_list[idx]}')
        axs[1, idx].set_ylabel('VIF')

        # Calculate eigenvalues and condition numbers for X'X matrix
        eigenvalues = []
        condition_numbers = []
        for i in range(1, len(df.columns) + 1):
            X_subset = df.iloc[:, 0:i]
            XTX = np.dot(X_subset.T, X_subset)
            eigenvalue = np.linalg.eigvals(XTX)
            eigenvalues.append(np.min(eigenvalue))
            condition_numbers.append(np.linalg.cond(XTX))

        # Create two separate axes for the condition number and minimum eigenvalue plot
        ax1 = axs[2, idx].twinx()
        ax2 = axs[2, idx].twinx()

        # Plot condition number on ax1
        ax1.spines['right'].set_position(('outward', 60))
        ax1.plot(range(1, len(df.columns) + 1), condition_numbers, color='red')
        ax1.set_ylabel('Condition Number', color='red')
        ax1.spines['left'].set_color('red')
        ax1.tick_params(axis='y', labelcolor='red')

        # Plot minimum eigenvalue on ax2
        ax2.plot(range(1, len(df.columns) + 1), eigenvalues, color='blue')
        ax2.set_ylabel('Minimum Eigenvalue', color='blue')
        ax2.spines['right'].set_color('blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim([np.min(eigenvalues), np.max(eigenvalues)])

        # Final plot settings
        axs[2, idx].set_xlabel('Number of Features')
        axs[2, idx].set_title(f'Minimum Eigenvalue and Condition Number - {title_list[idx]}')
        axs[2, idx].xaxis.set_major_locator(MultipleLocator(1))
        axs[2, idx].set_xlim([1, len(df.columns)])

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    
    # Show the plot
    plt.show()



######################################################################################################

def calculate_design_matrix(ds_neural_data, ds_behavior_data, bin_size=5):
    """
    Calculate the design matrix for a neural-behavioral data integration analysis.
    
    Args:
        ds_neural_data (pd.DataFrame): Downsampled neural data.
        ds_behavior_data (pd.DataFrame): Downsampled behavior data.
        bin_size (int): Number of time steps to use as the time kernel. For example, if the data is binned in 500 ms intervals,
                        a 2.5-second time kernel would be a bin size of 5 (2.5 sec / 0.5 sec). Default is 5.

    Returns:
        final_design_matrix (pd.DataFrame): A DataFrame containing the composite design matrix.
    """
    
    # Number of behavioral and neural variables
    number_of_behavioral_vars = ds_behavior_data.shape[1]
    no_neurons = ds_neural_data.shape[1]
    
    # Initialize an empty list to store each variable's time-lagged matrices
    matrices_list = []

    # Step 1: Generate a time-lagged matrix for each behavioral variable
    for j in range(number_of_behavioral_vars):
        variable_data = ds_behavior_data.iloc[:, j]
        
        # Initialize an empty matrix to hold the time-lagged data for this variable
        design_matrix = np.empty((ds_neural_data.shape[0] - bin_size + 1, bin_size))
        
        # Populate the design matrix with time-lagged data
        for start in range(ds_neural_data.shape[0] - bin_size + 1):
            temp_vector = variable_data.iloc[start:start + bin_size].values
            design_matrix[start, :] = temp_vector
        
        # Convert the design matrix for this variable to a DataFrame with labeled columns
        design_df = pd.DataFrame(design_matrix, columns=[f"{ds_behavior_data.columns[j]}_lag_{k+1}" for k in range(bin_size)])
        matrices_list.append(design_df)

    # Step 2: Concatenate all time-lagged matrices column-wise to form the final design matrix
    final_design_matrix = pd.concat(matrices_list, axis=1)

    return final_design_matrix



#############################################################################################################################
def model_fit(ds_neural_data, final_design_matrix, ds_behavior_data, bin_size=5, normalization_start=0, normalization_end=None):
    """
    This function regresses each neuron on the behavior variables and stores the beta coefficients and p-values.
    
    Args:
        ds_neural_data (pd.DataFrame): Downsampled neural data.
        final_design_matrix (pd.DataFrame): The final design matrix from calculate_design_matrix function.
        ds_behavior_data (pd.DataFrame): Downsampled behavior data with column names.
        bin_size (int): Time kernel in terms of bins. Must match bin size in calculate_design_matrix function. Default is 5.
        normalization_start (int): Row index for the start of normalization period. Default is 0.
        normalization_end (int or None): Row index for the end of normalization period. Default is None (entire session).
    
    Returns:
        beta_coefficients_matrix_final (pd.DataFrame): DataFrame containing beta coefficients.
        significance_matrix (pd.DataFrame): DataFrame containing p-values.
    """
    # Retrieve variable and neuron counts
    number_of_variables = ds_behavior_data.shape[1]
    num_neurons = ds_neural_data.shape[1]
    
    # Initialize matrices for storing results
    significance_matrix = np.empty((num_neurons, number_of_variables))
    beta_coefficients_matrix = np.empty((num_neurons, number_of_variables * bin_size + 2))  # +2 for intercept and autoregressive term

    # Loop over each neuron to fit regression models
    for neuron_k in range(num_neurons):
        print(f"Processing neuron {neuron_k + 1}/{num_neurons}")
        
        # Normalize neural trace to the specified baseline
        neural_trace = ds_neural_data.iloc[:, neuron_k]
        if normalization_end is None:
            normalization_end = len(neural_trace)
        neural_trace = (neural_trace - neural_trace[normalization_start:normalization_end].mean()) / neural_trace[normalization_start:normalization_end].std()

        # Prepare dependent variable and autoregressive term
        neural_trace_y = neural_trace.iloc[bin_size - 1:].values
        autoregressive_term = neural_trace.iloc[bin_size - 2:len(neural_trace) - 1].values
        
        # Combine autoregressive term with the design matrix
        final_matrix = pd.concat([pd.DataFrame({'Autoregressive_Term': autoregressive_term}), final_design_matrix], axis=1)
        
        # Initialize a list to store p-values for each variable
        p_values_vector = []

        # Loop over each variable, dropping columns to test significance
        for i in range(1, len(final_matrix.columns), bin_size):
            cols = list(range(i, i + bin_size))
            df_dropped_variable = final_matrix.drop(final_matrix.columns[cols], axis=1)
            
            # Fit the full and reduced models
            model_full = sm.OLS(neural_trace_y, sm.add_constant(final_matrix)).fit()
            model_reduced = sm.OLS(neural_trace_y, sm.add_constant(df_dropped_variable)).fit()
            
            # Perform ANOVA to compare models and get p-value
            full_vs_reduced = sm.stats.anova_lm(model_reduced, model_full)
            p_value = full_vs_reduced.iloc[1, -1]  # Assuming p-value is in the last column
            p_values_vector.append(p_value)

        # Store beta coefficients and p-values
        beta_coefficients_matrix[neuron_k, :] = model_full.params
        significance_matrix[neuron_k, :] = p_values_vector

    # Convert matrices to DataFrames with appropriate column names
    beta_coefficients_matrix_final = pd.DataFrame(beta_coefficients_matrix)
    variable_names = ds_behavior_data.columns
    final_names = [f"{var}_beta{k+1}" for var in variable_names for k in range(bin_size)]
    beta_column_names = ['Intercept', 'Autoregressive_Term'] + final_names
    beta_coefficients_matrix_final.columns = beta_column_names

    significance_matrix_df = pd.DataFrame(significance_matrix, columns=variable_names)

    return beta_coefficients_matrix_final, significance_matrix_df

