import joblib

def save_object(obj, filepath):
    """Saves a Python object to a file using joblib."""
    joblib.dump(obj, filepath)
    print(f"Object saved to {filepath}")

def load_object(filepath):
    """Loads a Python object from a file using joblib."""
    print(f"Loading object from {filepath}")
    return joblib.load(filepath)