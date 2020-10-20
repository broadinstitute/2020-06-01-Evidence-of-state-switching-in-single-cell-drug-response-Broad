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



def znormalization(data):
    
    
    # metadata = ['Image_FileName_OrigMito', 'Image_FileName_OrigER', 'Image_PathName_OrigER', 'Metadata_mg_per_ml',
    #      'Image_PathName_OrigDNA','Image_PathName_OrigRNA','Image_PathName_OrigMito','Image_FileName_OrigDNA',
    #      'Image_FileName_OrigRNA','Metadata_broad_sample','Image_Metadata_Well','Metadata_plate_map_name','Metadata_mmoles_per_liter',
    #      'Image_FileName_CellOutlines','Image_Metadata_Site','Metadata_Plate','Image_FileName_OrigAGP','Image_PathName_CellOutlines',
    #      'Image_PathName_OrigAGP','Cells_Location_Center_X','Cells_Location_Center_Y','Nuclei_Location_Center_X',
    #      'Nuclei_Location_Center_Y','Cytoplasm_Location_Center_X', 'Cytoplasm_Location_Center_Y','ObjectNumber']

    metadata = ['Metadata_Plate','Image_Metadata_Well']
    
    
    featlist = (data.columns[data.columns.str.contains('Cells_|Cytoplasm_|Nuclei_')]
               .tolist()
              )

    dt = data[featlist]
    
    
        
  ## Z normalized features
    
    scale = StandardScaler()

    scaled_data = scale.fit_transform(dt.values)
    
    dt = pd.DataFrame(scaled_data, columns= dt.columns)

    prf = data[metadata].merge(dt, how = left, left_index=True, right_index=True)
      
    
        
    return prf






def feature_selection(data):
    
    featlist = (data.columns[data.columns.str.contains('Cells_|Cytoplasm_|Nuclei_')]
               .tolist()
              )
    
    metadata = [col for col in data.columns if not col in featlist]

    dt = data[featlist]
    
    df= dt.dropna(axis=1, thresh= round(0.01 *len(dt)))
    
    ## Calculating VarianceThreshold

    var_th = VarianceThreshold(threshold=0.0)
    
    var_th.fit_transform(df.values)


    var_col = [col for col in df.columns if col not in df.columns[var_th.get_support()]]
    


    ##Correlation Threshold
    

    matrix = df.corr()
    
    
    corr_feat = set()
    
    for i in range(len(matrix.columns)):
        for j in range(i):
            if abs(matrix.iloc[i, j]) > 0.9:
                colname = matrix.columns[i]
                corr_feat.add(colname)
                
                
    # Removing features with less than 10 unique values
                
                
    counts = df.nunique()
    
    del_columns = [i for i, v in enumerate(counts) if v < (0.01 * len(df))]
    
    
    del_columns = df.columns[del_columns]

    blocklist_variables = list(df.columns[df.columns.str.contains('Location|Center|Parent|Count|Granularity_14|Granularity_15|Granularity_16|Manders|RWC|Costes')])

    remove_feat = list(set(corr_feat)) + var_col + blocklist_variables + list(set(del_columns))

    selected_features = set(df.columns) - set(remove_feat)
    
    prf = data[metadata].merge(data[selected_features], left_index=True, right_index=True)
    
    prf = prf.rename(columns = {'Image_Metadata_Well' : 'Metadata_Well'})
    
    
    return prf



# def whitening_transform(X, lambda_, rotate=True):
#     C = (1/X.shape[0]) * np.dot(X.T, X)
#     s, V = scipy.linalg.eigh(C)
#     D = np.diag( 1. / np.sqrt(s + lambda_) )
#     W = np.dot(V, D)
#     if rotate:
#         W = np.dot(W, V.T)
#     return W
    

# def whiten(X, mu, W):
#     return np.dot( X - mu, W)

# DO_WHITENING = True
# REG_PARAM = 1e-6



def IntersecOfSets(arr1, arr2, arr3, arr4, arr5): 
    # Converting the arrays into sets 
    s1 = set(arr1) 
    s2 = set(arr2) 
    s3 = set(arr3)
    s4 = set(arr4) 
    s5 = set(arr5) 
      
    # Calculates intersection of  
    # sets on s1 through s5
    set1 = s1.intersection(s2)
      
    # Calculates intersection of sets 
    # on set1 and s3 
    set2 = set1.intersection(s3)
    
    set3 = set2.intersection(s4) 
    
    result_set = set3.intersection(s4) 
      
    # Converts resulting set to list 
    final_list = list(result_set) 
    
    return final_list


        
        
def whiten(X, method):

        
         ## Calculating Covariance matrix
        
    sigma = (1/X.shape[0]) * np.dot(X.T, X)
    
    ## Eigen vectors and Eigen values
    
    evals, evecs = np.linalg.eigh(sigma)
    
    if method == "ZCA":
        
        # Calculating whitening matrix

        W_matrix = np.dot(np.dot(evecs,np.diag(1.0/ np.sqrt(evals + 1e-6))), evecs.T)
  
    
        # Transforming centered data to whitened matrix

        whitened = np.dot((X - X.mean()), W_matrix)
        
        
    elif method == "PCA":
        

        W_matrix = np.dot(np.diag(1.0/ np.sqrt(evals + 1e-6)), evecs)
        
        whitened = np.dot((X - X.mean()), W_matrix)

    return whitened



        


    