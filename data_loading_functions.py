
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
