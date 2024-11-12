def format_clustering_parameter_dict_to_text(input_dict):
    result_str = ""
    for metric, measures in input_dict.items():
        result_str += f"{metric}\n"
        for measure, value in measures.items():
            result_str += f"{measure}: {value}    "  # Replace tabs with spaces for consistent display
        result_str += "\n"  # Add a new line after each metric
    
    # delete last \n
    result_str = result_str[:-2]
    return result_str
