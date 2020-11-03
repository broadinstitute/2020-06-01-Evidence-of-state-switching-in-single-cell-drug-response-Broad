#!/usr/bin/env python
# coding: utf-8

# ## Apply Harmony batch effect correction
# 
# Here, we apply Harmony in two distinct ways:
# 
# 1. Per plate harmony using single cell normalized profiles adjusting for Well
# 2. All plate harmony using concatenated single cell normalized profiles adjusting for Plate and Well
# 
# We use the [`harmonypy`](https://github.com/slowkow/harmonypy) implementation (version 0.0.5) of the Harmony algorithm introduced in [Korsunsky et al. 2019](https://doi.org/10.1038/s41592-019-0619-0)

# In[1]:


import pathlib
import numpy as np
import pandas as pd

import harmonypy as hm
from sklearn.decomposition import PCA

from pycytominer.cyto_utils import infer_cp_features


# In[2]:


np.random.seed(123)


# In[3]:


# Load constants
data_dir = pathlib.Path("../data")
feature_select_summary_file = pathlib.Path("tables/feature_select_summary.csv")
output_file_suffix = "dmso_harmony_normalized.csv"
output_file_inverse_suffix = "dmso_inverse_harmony_normalized.csv"
minimum_times_selected = 2
num_pcs = 20

harmony_random_state = 0
harmony_adjust_vars_perplate = ["Image_Metadata_Well"]
harmony_adjust_vars_full = ["Image_Metadata_Well", "Metadata_Plate"]


# In[4]:


# Load data
data_dir = pathlib.Path("../data")
data_files = [x for x in data_dir.iterdir() if "_dmso_normalized.csv" in str(x)]
data_files


# In[5]:


# Perform feature selection
feature_df = pd.read_csv(feature_select_summary_file, index_col=0)

feature_df = (
    feature_df
    .assign(times_selected=feature_df.sum(axis="columns"))
    .assign(selected=0)
)

# Load feature selection options
feature_df.loc[feature_df.times_selected > minimum_times_selected, "selected"] = 1
feature_df = feature_df.query("selected == 1")

print(feature_df.shape)
feature_df.head()


# In[6]:


# Perform Harmony normalization per plate
all_dfs = []
for file in data_files:
    # Extract plate from file name
    plate = str(file).split("/")[-1].split("_")[0]
    print(f"Now processing {plate}...")
    
    # Load data and apply feature selection
    df = pd.read_csv(file).reindex(feature_df.index, axis="columns")
    
    # Extract feature types
    morphology_features = infer_cp_features(df, compartments=["Cells", "Cytoplasm", "Nuclei"])
    metadata_cols = ["Image_Metadata_Well"] + infer_cp_features(df, metadata=True)
    
    # Fit PCA
    pca = PCA(n_components=num_pcs)
    pca.fit(df.loc[:, morphology_features])
    
    # Transform PCA
    pc_df = pca.transform(df.loc[:, morphology_features])
    pc_df = pd.DataFrame(pc_df).add_prefix("pca_") 
    
    # Apply harmony per plate
    harmony_out = (
        hm.run_harmony(
            data_mat=pc_df,
            meta_data=df.loc[:, metadata_cols],
            vars_use=harmony_adjust_vars_perplate,
            random_state=harmony_random_state
        )
    )
    
    # Compile harmony output file
    harmony_df = pd.concat(
        [
            df.loc[:, metadata_cols],
            pd.DataFrame(harmony_out.Z_corr).transpose().add_prefix("harmonized_")
        ],
        axis="columns"
    )

    # Output harmonized file
    output_file = pathlib.Path(f"{data_dir}/{plate}_{output_file_suffix}")
    harmony_df.to_csv(output_file, index=False, sep=",")

    # Apply an inverse transform to get back to original feature space
    inverse_harmony_df = pd.concat(
        [
            df.loc[:, metadata_cols],
            pd.DataFrame(pca.inverse_transform(harmony_out.Z_corr.transpose()), columns=morphology_features)
        ],
        axis="columns"
    )
    
    # Output inverse harmonized file
    output_file = pathlib.Path(f"{data_dir}/{plate}_{output_file_inverse_suffix}")
    inverse_harmony_df.to_csv(output_file, index=False, sep=",")

    # Append all cells to all dfs
    all_dfs.append(df)
    del df
    del harmony_df
    del inverse_harmony_df


# In[7]:


# Apply Harmony normalization for all plates at once
all_dfs = pd.concat(all_dfs, axis="rows").reset_index(drop=True)

# Fit PCA
pca = PCA(n_components=num_pcs)
pca.fit(all_dfs.loc[:, morphology_features])

# Transform PCA
pc_df = pca.transform(all_dfs.loc[:, morphology_features])
pc_df = pd.DataFrame(pc_df).add_prefix("pca_") 

# Apply harmony per plate
harmony_out = (
    hm.run_harmony(
        data_mat=pc_df,
        meta_data=all_dfs.loc[:, metadata_cols],
        vars_use=harmony_adjust_vars_full,
        random_state=harmony_random_state
    )
)

# Compile harmony output file
harmony_df = pd.concat(
    [
        all_dfs.loc[:, metadata_cols],
        pd.DataFrame(harmony_out.Z_corr).transpose().add_prefix("harmonized_all_plates_")
    ],
    axis="columns"
)

# Output harmonized file
output_file = pathlib.Path(f"{data_dir}/dmso_harmonized_all_plates.csv.gz")
harmony_df.to_csv(output_file, index=False, sep=",", compression="gzip")

print(harmony_df.shape)
harmony_df.head(2)


# In[8]:


# Apply an inverse transform to get back to original feature space
inverse_harmony_df = pd.concat(
    [
        all_dfs.loc[:, metadata_cols],
        pd.DataFrame(pca.inverse_transform(harmony_out.Z_corr.transpose()), columns=morphology_features)
    ],
    axis="columns"
)

# Output inverse harmonized file
output_file = pathlib.Path(f"{data_dir}/dmso_inverse_harmonized_all_plates.csv.gz")
inverse_harmony_df.to_csv(output_file, index=False, sep=",")

print(inverse_harmony_df.shape)
inverse_harmony_df.head(2)

