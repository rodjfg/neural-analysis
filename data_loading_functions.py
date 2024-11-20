import pandas as pd

def read_behavior_data(file_path):
    """
    This function will load CSV files from ANYMAZE

    Read behavior data from a CSV file and select relevant columns.

    Args:
    - file_path (str): The file path of the CSV file containing behavior data.

    Returns:
    - behavior_data (pd.DataFrame): A Pandas DataFrame containing the selected behavior data.
    """

    behavior_data = pd.read_csv(file_path, encoding= 'unicode_escape', header=0,)

    # rename first column to 'Time'
    # This will help keep the name consistent for the time column to be called in 
    # later functions, like when downsampling. 
    behavior_data = behavior_data.rename(columns={behavior_data.columns[0]: 'Time'})

    return behavior_data

#To load neural data
def read_neural_data(file_path):
    """
    Read neural data from a CSV file, filtering to only include accepted cells.

    Args:
    - file_path (str): The file path of the CSV file containing neural data.

    Returns:
    - neural_data (pd.DataFrame): A Pandas DataFrame containing the neural data.
    """
    # Read CSV file, skipping first row and using second row as header
    neural_data = pd.read_csv(file_path, skiprows=[1], header=0)
    neural_data = neural_data.rename(columns={neural_data.columns[0]: 'Time'})

    # Read first row to determine accepted vs rejected cells
    cell_status = pd.read_csv(file_path, nrows=1)

    # Select only accepted cells
    bool_status = cell_status.iloc[0].str.contains('accepted')
    accepted_cols = bool_status[bool_status == True].index.tolist()
    accepted_cols.insert(0, 'Time')

    # Filter df to include only accepted cells
    neural_data = neural_data[accepted_cols]

    return neural_data

