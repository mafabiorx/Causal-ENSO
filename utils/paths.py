import os
from pathlib import Path

# Project root directory - the directory containing this file's parent directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

# Results directories
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Function to create directories if they don't exist
def create_directories():
    """Create all necessary directories if they don't exist."""
    for directory in [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        FIGURES_DIR, MODELS_DIR, TABLES_DIR
    ]:
        os.makedirs(directory, exist_ok=True)

# Function to get path for a specific data file
def get_data_path(file_name, data_type="processed"):
    """
    Get the full path for a data file.
    
    Parameters:
    -----------
    file_name : str
        Name of the file
    data_type : str
        Type of data: "raw", "interim", "processed", or "external"
        
    Returns:
    --------
    str
        Full path to the file
    """
    if data_type == "raw":
        return os.path.join(RAW_DATA_DIR, file_name)
    elif data_type == "interim":
        return os.path.join(INTERIM_DATA_DIR, file_name)
    elif data_type == "processed":
        return os.path.join(PROCESSED_DATA_DIR, file_name)
    elif data_type == "external":
        return os.path.join(EXTERNAL_DATA_DIR, file_name)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# Function to get path for a results file
def get_results_path(file_name, result_type="figures"):
    """
    Get the full path for a results file.
    
    Parameters:
    -----------
    file_name : str
        Name of the file
    result_type : str
        Type of result: "figures", "models", or "tables"
        
    Returns:
    --------
    str
        Full path to the file
    """
    if result_type == "figures":
        return os.path.join(FIGURES_DIR, file_name)
    elif result_type == "models":
        return os.path.join(MODELS_DIR, file_name)
    elif result_type == "tables":
        return os.path.join(TABLES_DIR, file_name)
    else:
        raise ValueError(f"Unknown result type: {result_type}")

# Create all directories on import
create_directories()