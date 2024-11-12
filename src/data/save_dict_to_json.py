import json
import os
import numpy as np

def convert_numpy_types(obj):
    """
    Recursively converts numpy types to native Python types in a dictionary.
    
    :param obj: The object to convert
    :return: The converted object
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    else:
        return obj

def save_dict_to_json(file_path, dict_data):
    """
    Saves a dictionary to a JSON file at the specified path, ensuring that numpy types are converted.
    
    :param file_path: The path to the file to save the dictionary to
    :param dict_data: The dictionary to save to the file
    """
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert any numpy types to native Python types
    dict_data = convert_numpy_types(dict_data)

    # Save the dictionary to the file as JSON
    with open(file_path, 'w') as json_file:
        json.dump(dict_data, json_file, indent=4)
    
    print(f"Dictionary saved to {file_path}")
