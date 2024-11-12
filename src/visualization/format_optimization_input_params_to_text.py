import numpy as np

def format_optimization_input_params_to_text(**kwargs):
    result_str = ""
    param_items = list(kwargs.items())  # Convert dict items to list for index checking
    
    for i, (param_name, param_value) in enumerate(param_items):
        if isinstance(param_value, range):  # Check if it's a range
            result_str += f"{param_name}: {param_value.start} - {param_value.stop - 1}"
        elif isinstance(param_value, np.ndarray):  # Handle np.arange (numpy arrays)
            if len(param_value) > 1:  # Check if array has more than one element
                result_str += f"{param_name}: {param_value[0]:.2f} - {param_value[-1]:.2f} (step: {param_value[1] - param_value[0]:.2f})"
            else:  # Handle case where there's only one element
                result_str += f"{param_name}: {param_value[0]:.2f} (single value)"
        elif isinstance(param_value, list) and all(isinstance(item, str) for item in param_value):  # Handle list of strings
            result_str += f"{param_name}: " + ", ".join(param_value)
        elif isinstance(param_value, (str, float, int)):  # Handle scalar values like metric or xi
            result_str += f"{param_name}: {param_value}"
        else:
            result_str += f"{param_name}: Unknown type"
        
        # Only add a newline if it's not the last item
        if i < len(param_items) - 1:
            result_str += "\n"
    
    return result_str
