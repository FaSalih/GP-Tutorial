import pandas as pd
import numpy as np
import gpflow
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib, os, argparse, warnings, pickle
from tqdm import tqdm

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : np array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
                           zero; uses a small buffer,
                           (log(x+b)-mean(log(x+b)))/std(log(x+b))
            . Sqrt - standardization on the sqrt of scaled variables using k=10,
                           (sqrt(x/k)-mean(sqrt(x/k)))/std(sqrt(x/k))
        Defalt: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : np array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # Convert input to numpy array if not on already
    inputArray=np.array(inputArray)
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If method is None, return inputArray as it is
    if method is None: return inputArray, None
    # Set bias that prevent taking log o sqrt of negative numbers (abs(bias) should be > abs(min(inputArray)))
    bias=10
    # If skScaler is None and in forward mode, train scaler for the first time
    if skScaler is None:
        # Check scaling method
        if method=='Standardization' or method=='MinMax': 
            aux=inputArray
        elif method=='LogStand': 
            aux=np.log(inputArray)
        elif method=='Log+bStand': 
            # Get bias from inputArray itself (prevent taking log of a negative number)
            aux=np.log(inputArray+bias)
        elif method=='Sqrt': 
            aux=np.sqrt((inputArray+bias)/100)
        else: raise ValueError('Could not recognize method in normalize().')
        # Normalize scaled data
        if method not in ['MinMax', 'Sqrt']:
            skScaler=StandardScaler().fit(aux)
        else:
            skScaler=MinMaxScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand': inputArray=np.exp(inputArray)
        elif method=='Log+bStand': inputArray=np.exp(inputArray)-bias
        elif method=='Sqrt': inputArray=100*inputArray**2 - bias
    elif not reverse:
        # Check method and scale un-normalized data if necessary
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=np.log(inputArray)
        elif method=='Log+bStand': aux=np.log(inputArray+bias)
        elif method=='Sqrt': aux=np.sqrt((inputArray+bias)/100)
        else: raise ValueError('Could not recognize method in normalize().')
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

