def create_three_letter_uid_for_optics_optimization_init_params(max_eps_range, min_samples_range, xi_range):
    """
    Create a 3-letter unique ID based on optimization parameters.
    """
    # Take the first letter of 'max_eps', 'min_samples', and 'xi' ranges for uniqueness
    max_eps_id = chr(65 + int(max(max_eps_range)) % 26)  # Convert max_eps to a letter
    min_samples_id = chr(65 + int(min(min_samples_range)) % 26)  # Convert min_samples to a letter
    xi_id = chr(65 + int(min(xi_range) * 100) % 26)  # Convert xi to a letter based on its percentage

    # Combine them to form a 3-letter ID
    uid = f"{max_eps_id}{min_samples_id}{xi_id}"
    return uid