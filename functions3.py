import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_diagnostics_analysis(df_list, title_list):
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'

    fig, axs = plt.subplots(3, 5, figsize=(55, 24), gridspec_kw={'hspace': 0.8, 'wspace': 0.5})

    for idx, df in enumerate(df_list):
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[0, idx])
        axs[0, idx].set_title('Correlation Heatmap')

        X = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
        vif = pd.DataFrame()
        vif["variables"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.sort_values("VIF", ascending=False).plot(x="variables", y="VIF", kind='bar', legend=False, ax=axs[1, idx])
        axs[1, idx].set_title('VIF values')
        axs[1, idx].set_ylabel('VIF')

        eigenvalues = []
        condition_numbers = []
        for i in range(1, len(df.columns) + 1):
            X_subset = df.iloc[:, 0:i]
            XTX = np.dot(X_subset.T, X_subset)
            eigenvalue = np.linalg.eigvals(XTX)
            eigenvalues.append(np.min(eigenvalue))
            condition_numbers.append(np.linalg.cond(XTX))

        ax1 = axs[2, idx].twinx()
        ax2 = axs[2, idx].twinx()

        ax1.spines['right'].set_position(('outward', 60))
        ax1.yaxis.set_label_position("left")
        ax1.yaxis.tick_left()

        ax1.plot(range(1, len(df.columns) + 1), condition_numbers, color='red')
        ax1.set_ylabel('Condition Number', color='red')
        ax1.spines['left'].set_color('red')
        ax1.spines['left'].set_linewidth(0.5)
        ax1.tick_params(axis='y', labelcolor='red')

        axs[2, idx].spines['left'].set_visible(False)
        axs[2, idx].yaxis.set_visible(False)

        ax2.plot(range(1, len(df.columns) + 1), eigenvalues, color='blue')
        ax2.set_ylabel('Minimum Eigenvalue', color='blue')
        ax2.spines['right'].set_color('blue')
        ax2.spines['right'].set_linewidth(0.5)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim([np.min(eigenvalues), np.max(eigenvalues)])

        axs[2, idx].set_xlabel('Number of features')
        axs[2, idx].set_title('Minimum Eigenvalue and Condition Number of X\'X')
        axs[2, idx].xaxis.set_major_locator(MultipleLocator(1))
        axs[2, idx].set_xlim([1, len(df.columns)])

    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.show()

# Example usage (when you want to use the function):
# df_list = [beh_data_M1_D1, beh_data_M2_D1, ...]
# title_list = ['Mouse 1', 'Mouse 2', ...]
# plot_diagnostics_analysis(df_list, title_list)

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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def clean_string(s):
    s = s.replace('_', ' ')  # replace underscores with spaces
    s = s.title()  # capitalize the first letter of each word
    return s
