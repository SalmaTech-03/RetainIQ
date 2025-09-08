import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a specified CSV file path."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None