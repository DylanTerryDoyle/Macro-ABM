import yaml
import numpy as np
import powerlaw as pl
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

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
