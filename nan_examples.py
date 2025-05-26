"""
Test file demonstrating NaN equivalence comparison issues.
This file contains examples of both correct and incorrect NaN handling.
"""
import numpy as np
import pandas as pd
import math

def bad_nan_comparison():
    """Example of incorrect NaN comparison - should trigger pylint warning."""
    data = np.array([1.0, 2.0, np.nan, 4.0])
    
    # This is WRONG - will trigger our custom pylint rule
    mask = data == np.nan  # W9901: nan-equivalence-misuse
    
    # This is also WRONG
    if data[2] != np.nan:  # W9901: nan-equivalence-misuse
        print("Not NaN")
    
    return mask

def good_nan_comparison():
    """Example of correct NaN comparison."""
    data = np.array([1.0, 2.0, np.nan, 4.0])
    
    # This is CORRECT
    mask = np.isnan(data)
    
    # This is also CORRECT
    if np.isnan(data[2]):
        print("Is NaN")
    
    return mask

def bad_pandas_nan_comparison():
    """Example of incorrect NaN comparison with pandas."""
    df = pd.DataFrame({'values': [1.0, 2.0, np.nan, 4.0]})
    
    # This is WRONG
    mask = df['values'] == np.nan  # W9901: nan-equivalence-misuse
    
    return mask

def good_pandas_nan_comparison():
    """Example of correct NaN comparison with pandas."""
    df = pd.DataFrame({'values': [1.0, 2.0, np.nan, 4.0]})
    
    # These are CORRECT
    mask1 = pd.isna(df['values'])
    mask2 = df['values'].isna()
    mask3 = pd.isnull(df['values'])
    
    return mask1, mask2, mask3

def mixed_nan_operations():
    """Example mixing good and bad practices."""
    data = [1.0, float('nan'), 3.0]
    
    # This is WRONG
    problematic = [x for x in data if x != float('nan')]  # Should trigger warning
    
    # This is CORRECT
    clean_data = [x for x in data if not math.isnan(x)]
    
    return clean_data
