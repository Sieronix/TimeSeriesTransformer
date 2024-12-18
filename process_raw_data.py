import pandas as pd
from src.utils.debug_utils import LOG_DEBUG_PRINT


# -----------------------------------------------------------------------------------------
# Configuration Parameters
DEBUG_MODE = False            # Enable or disable debug mode for logging
ma_window_16 = 16             # Window size for the 16-period moving average
ma_window_32 = 32             # Window size for the 32-period moving average
start_index = 0               # Starting index for data filtering
end_index = 500000            # Ending index for data filtering
# -----------------------------------------------------------------------------------------


def load_pickle_data(file_path):
    """
    Loads data from a generic pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the pickle file.
    """
    raw_data = pd.read_pickle(file_path)
    if DEBUG_MODE:
        LOG_DEBUG_PRINT("RAW DATA", ("raw_data", raw_data))
    return raw_data


def select_columns(raw_data):
    """
    Selects specific columns from the raw data for further processing.

    Args:
        raw_data (pd.DataFrame): The raw data.

    Returns:
        pd.DataFrame: DataFrame containing only the selected columns.
    """
    selected_data = raw_data[['close_price']].copy()
    if DEBUG_MODE:
        LOG_DEBUG_PRINT("SELECTED COLUMNS", ("selected_data", selected_data))
    return selected_data


def create_moving_average_columns(data):
    """
    Adds moving average columns to the data.

    Args:
        data (pd.DataFrame): The data with the selected columns.

    Returns:
        pd.DataFrame: DataFrame with added moving average columns.
    """
    data['ma_16'] = data['close_price'].rolling(window=ma_window_16).mean()
    data['ma_32'] = data['close_price'].rolling(window=ma_window_32).mean()
    if DEBUG_MODE:
        LOG_DEBUG_PRINT("WITH MOVING AVERAGE", ("data_with_ma", data))
    return data


def filter_by_range(data):
    """
    Filters the data to include only rows within a specified index range.

    Args:
        data (pd.DataFrame): The data with moving averages.

    Returns:
        pd.DataFrame: DataFrame filtered by the specified index range.
    """
    ranged_data = data.iloc[start_index:end_index+1]
    if DEBUG_MODE:
        LOG_DEBUG_PRINT("FILTERED BY RANGE", ("ranged_data", ranged_data))
    return ranged_data


def process_raw_data_main(file_path):
    """
    Main function for processing raw data.

    Workflow:
        1. Load raw data from a generic pickle file.
        2. Select specific columns for analysis.
        3. Compute moving average columns.
        4. Filter the data by the specified range.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: Final processed data ready for further analysis.
    """
    raw_data = load_pickle_data(file_path)
    selected_data = select_columns(raw_data)
    data_with_ma = create_moving_average_columns(selected_data)
    final_data = filter_by_range(data_with_ma)
    if DEBUG_MODE:
        LOG_DEBUG_PRINT("FINAL DATA", ("final_data", final_data))
    return final_data
