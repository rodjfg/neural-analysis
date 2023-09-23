import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
!pip install num2words

def downsample_behavior_data(behavior_data, frequency):
    """
    Downsample behavior data to a specified frequency.
    
    Args:
    - behavior_data (pd.DataFrame): A Pandas DataFrame containing the behavior data.
    - frequency (str): The frequency to downsample to, in Pandas resample format (e.g. '500ms').
    
    Returns:
    - behavior_data_ds (pd.DataFrame): A Pandas DataFrame containing the downsampled behavior data.
    """
    
    # Set the index once outside the loop
    behavior_data = behavior_data.set_index(pd.to_timedelta(behavior_data['Time'], unit='s'))
    list_of_column_names = list(behavior_data.columns)
    behavior_data_ds = pd.DataFrame()
    
    for column in list_of_column_names:
        if column == 'Time':
            continue  # Skip the 'Time' column
        elif column in ['In platform', 'In REWARD ZONE', 'In Center']:
            output = behavior_data[column].resample(frequency).last()
        else:
            output = behavior_data[column].resample(frequency).mean()
        
        if column in ['Tone', 'Shock']:
            output.fillna(0, inplace=True)
        else:
            output.fillna(method='bfill', inplace=True)
        
        # Reset index to total seconds
        output.index = output.index.total_seconds()
        behavior_data_ds[column] = output
        
    return behavior_data_ds




def calculate_matrices(ds_neural_data, behavior_data_ds, bin_size=5):
    neural_trace = ds_neural_data.iloc[:,0]
    data = behavior_data_ds
    features = data.shape[1]
    no_neurons = ds_neural_data.shape[1]
    design_matrix = np.empty(shape=(len(neural_trace)-bin_size+1, bin_size))
    significance_matrix = np.empty(shape=(no_neurons, features))
    variable_names = data.columns
    beta_coefficients_matrix = np.empty(shape=(no_neurons, features*bin_size+1+1))
    matrices_list = []
    
    for j in range(features):
        names = [variable_names[j] + ' beta' + str(k+1) for k in range(bin_size)]
        variable_n = data.iloc[:,j]
        for start in range(len(neural_trace)-bin_size+1):
            temp_vector = variable_n.iloc[start:start+bin_size]
            design_matrix[start,:] = temp_vector
        matrices_list.append(design_matrix.copy())
    
    for jj in range(len(variable_names)):
        matrices_list[jj] = pd.DataFrame(matrices_list[jj])
        matrices_list[jj].columns = [variable_names[jj] + ' beta' + str(k+1) for k in range(bin_size)]
    
    final_matrix = pd.concat(matrices_list, axis=1)
    
    return final_matrix




import numpy as np
import pandas as pd
import statsmodels.api as sm

def model_fit(ds_neural_data, final_matrix, bin_size, features, start=0):
    """
    Process neural data and compute significance and beta coefficients.

    Args:
    - ds_neural_data (pd.DataFrame): The neural data.
    - final_matrix (pd.DataFrame): The final matrix.
    - bin_size (int): The bin size.
    - features (int): The number of features.
    - start (int, optional): The start index. Default is 0.

    Returns:
    - beta_coefficients_matrix (np.array): The beta coefficients matrix.
    - significance_matrix (np.array): The significance matrix.
    """
    
    num_neurons = ds_neural_data.shape[1]
    significance_matrix = np.empty(shape=(num_neurons, len(final_matrix.columns) // bin_size))
    beta_coefficients_matrix = np.empty(shape=(num_neurons, features*bin_size + 2))  # +1 for ARX

    for neuron_k in range(num_neurons):
        print(neuron_k)
        neural_trace = ds_neural_data.iloc[:, neuron_k]
        neural_trace = (neural_trace - np.mean(neural_trace[:300])) / np.std(neural_trace[:300])  # normalize to 1st 5 min
        
        y_variable = neural_trace.iloc[(start + bin_size - 1):]
        y_variable = list(y_variable)
        
        ARX = neural_trace.iloc[start + bin_size - 2:len(neural_trace) - 1]
        ARX = list(ARX)
        
        final_matrix_ARX = pd.concat([pd.DataFrame({'ARX': ARX}), final_matrix], axis=1)
        p_values_vector = []
        for i in range(1, len(final_matrix_ARX.columns), bin_size):
            cols = list(range(i, i + bin_size))
            df_dropped_variable = final_matrix_ARX.drop(final_matrix_ARX.columns[cols], axis=1)
            model_1 = sm.OLS(y_variable, sm.add_constant(final_matrix_ARX)).fit()
            model_2 = sm.OLS(y_variable, sm.add_constant(df_dropped_variable)).fit()
            
            full_vs_reduced = sm.stats.anova_lm(model_2, model_1)
            pvalue = full_vs_reduced.iloc[1, 5]
            p_values_vector.append(pvalue)

        model_1 = sm.OLS(y_variable, sm.add_constant(final_matrix_ARX)).fit()
        beta_coefficients_matrix[neuron_k, :] = model_1.params
        significance_matrix[neuron_k, :] = p_values_vector
    
    beta_coefficients_matrix_final=pd.DataFrame(data=beta_coefficients_matrix)
    significance_matrix = pd.DataFrame(data=significance_matrix)#, columns=variable_names)



    return beta_coefficients_matrix_final, significance_matrix





import numpy as np
import num2words
from num2words import num2words
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def number_to_words(n):
    return num2words(n)

def clean_string(s):
    s = s.replace('_', ' ')  # replace underscores with spaces
    s = s.title()  # capitalize the first letter of each word
    return s
