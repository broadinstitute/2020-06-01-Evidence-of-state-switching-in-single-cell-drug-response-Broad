#!/usr/bin/env python
# coding: utf-8

# ## Apply normalization and feature selection to single cell DMSO profiles

# In[1]:


import pathlib
import numpy as np
import pandas as pd

from pycytominer import normalize
from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Load constants
data_dir = pathlib.Path("../data")

feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
]

feature_select_summary_file = pathlib.Path("tables/feature_select_summary.csv")


# In[3]:


# Load data
data_dir = pathlib.Path("../data")
data_files = [x for x in data_dir.iterdir() if "_dmso.csv" in str(x)]
data_files


# In[4]:


features_selected_list = []
for file in data_files:
    plate = str(file).split("/")[-1].split("_")[0]
    
    print(f"Now processing {plate}...")
    df = pd.read_csv(file)

    metadata_cols = ["Image_Metadata_Well"] + infer_cp_features(df, metadata=True)
    feature_cols = infer_cp_features(df, compartments=["Cells", "Cytoplasm", "Nuclei"])

    output_file = pathlib.Path(f"{data_dir}/{plate}_dmso_normalized.csv")
    
    # Apply normalization and output files
    normalize(
        profiles=df,
        features=feature_cols,
        meta_features=metadata_cols,
        method="standardize",
        output_file=output_file
    )
    
    # Apply feature selection only to determine which features to use
    feature_select_df = feature_select(
        profiles=output_file,
        features="infer",
        operation=feature_select_ops,
        na_cutoff=0,
        corr_threshold=0.8
    )
    
    # Identify which features were selected
    selected_features = pd.DataFrame(np.zeros((len(df.columns), 1)), index=df.columns, columns=[plate])
    selected_features.loc[selected_features.index.isin(feature_select_df.columns), plate] = 1
    selected_features = selected_features.astype(int)
    
    features_selected_list.append(selected_features)


# In[5]:


feature_select_summary_df = pd.concat(features_selected_list, axis="columns")

feature_select_summary_df.to_csv(feature_select_summary_file, sep=",", index=True)

print(feature_select_summary_df.shape)
feature_select_summary_df.head()

