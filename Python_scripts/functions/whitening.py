import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotnine as gg
import umap
from pathlib import Path
import scipy.linalg
from sklearn.feature_selection import VarianceThreshold




class whitening_transform:

    # 'https://cbrnr.github.io/2018/12/17/whitening-pca-zca/'

    
    def __init__(self, X, method, REG_PARAM = 1e-6):
        
        self.method = method
        self.X = X
        self.REG_PARAM = REG_PARAM
        
        
    def whiten(self):
        
        
         ## Calculating Covariance matrix
        
        sigma = (1/self.X.shape[0]) * np.dot(self.X.T, self.X)
        
        ## Eigen vectors and Eigen values
        
        evals, evecs = np.linalg.eigh(sigma)
        
        if self.method == "ZCA":
            
            # Calculating whitening matrix

            W_matrix = np.dot(np.dot(evecs,np.diag(1.0/ np.sqrt(evals + self.REG_PARAM))), evecs.T)
      
        
            # Transforming centered data to whitened matrix

            whitened = np.dot((self.X - self.X.mean()), W_matrix)
            
            
        elif self.method == "PCA":
            

            W_matrix = np.dot(np.diag(1.0/ np.sqrt(evals + self.REG_PARAM)), evecs)
            
            whitened = np.dot((self.X - self.X.mean()), W_matrix)
    
        return whitened
