def format_string(input_str: str) -> str:
    """
    将输入字符串中的 'unspecified' 替换为 'multiple'
    
    Args:
        input_str (str): 需要格式化的输入字符串
        
    Returns:
        str: 格式化后的字符串
    """
    return input_str.replace('unspecified', 'multiple')
