"""
Helper functions for MaterialVision project.
This module contains utility functions for data processing, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import json
import re

def list_to_image(img_list, size=224):
    """
    Convert a JSON list to a 2D image.
    
    Args:
        img_list (str): JSON string containing image data
        size (int): Size of the output image (default: 224)
        
    Returns:
        np.ndarray: 2D numpy array representing the image
    """
    return np.array(json.loads(img_list)).reshape(size, size)

def extract_formula_bandgap(text):
    """
    Extract chemical formula and bandgap value from text.
    
    Args:
        text (str): Input text containing formula and bandgap information
        
    Returns:
        Tuple[str, float]: Chemical formula and bandgap value
    """
    formula_match = re.search(r'The chemical formula is ([A-Za-z0-9]+)', text)
    bandgap_match = re.search(r'mbj_bandgap value is ([0-9.]+)', text)
    formula = formula_match.group(1) if formula_match else None
    bandgap_str = bandgap_match.group(1).rstrip('.') if bandgap_match else None  # strip trailing dot
    bandgap = float(bandgap_str) if bandgap_str else None 
    return formula, bandgap

