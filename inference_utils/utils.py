def format_string(input_str: str) -> str:
    """
    Replace 'unspecified' with 'multiple' in the input string.
    
    Args:
        input_str (str): The input string to format.
        
    Returns:
        str: The formatted string.
    """
    return input_str.replace('unspecified', 'multiple')
