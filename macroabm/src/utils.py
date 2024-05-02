import yaml

def load_parameters(filename: str):
    """
    Load YAML file as dictionary.
    
    Parameters
    ----------
        filename : str
            name of YAML file to load
    
    Returns
    -------
        file_dict : dict
            YAML file loaded as dictionary 
    """
    with open(filename, 'r') as file:
        file_dict = yaml.safe_load(file)
    return file_dict