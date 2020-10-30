#!/usr/bin/env python
# coding: utf-8

# ## Apply spherize transform
# 
# Spherize (aka whiten) the normalized profiles using ZCA-cor transform

# In[1]:


import pathlib
import numpy as np
import pandas as pd

from pycytominer import normalize
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Load constants
data_dir = pathlib.Path("../data")
feature_select_summary_file = pathlib.Path("tables/feature_select_summary.csv")
output_file_suffix = "dmso_ZCA-cor_normalized.csv"
minimum_times_selected = 2


# In[3]:


# Load data
data_dir = pathlib.Path("../data")
data_files = [x for x in data_dir.iterdir() if "_dmso_normalized.csv" in str(x)]
data_files


# In[4]:


# Perform feature selection
feature_df = pd.read_csv(feature_select_summary_file, index_col=0)

feature_df = (
    feature_df
    .assign(times_selected=feature_df.sum(axis="columns"))
    .assign(selected=0)
)

feature_df.times_selected.value_counts()


# In[5]:


# Load feature selection options
feature_df.loc[feature_df.times_selected > minimum_times_selected, "selected"] = 1
feature_df = feature_df.query("selected == 1")

print(feature_df.shape)
feature_df.head()


# In[6]:


# Perform spherize transform
for file in data_files:
    # Extract plate from file name
    plate = str(file).split("/")[-1].split("_")[0]
    print(f"Now processing {plate}...")
    
    # Load data and apply feature selection
    df = pd.read_csv(file).reindex(feature_df.index, axis="columns")

    # Get feature names
    metadata_cols = ["Image_Metadata_Well"] + infer_cp_features(df, metadata=True)
    feature_cols = infer_cp_features(df, compartments=["Cells", "Cytoplasm", "Nuclei"])

    output_file = pathlib.Path(f"{data_dir}/{plate}_{output_file_suffix}")
    
    # Apply spherize transformation and output files
    normalize(
        profiles=df,
        features=feature_cols,
        meta_features=metadata_cols,
        method="spherize",
        spherize_method="ZCA-cor",
        spherize_center=True,
        output_file=output_file
    )

