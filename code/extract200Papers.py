import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import pdfplumber
import os
from openai import OpenAI
import shutil    

df_feature4 = pd.read_csv("./basic_features_v4.csv")
df_rct = pd.read_csv("./rct_patients.csv")
merged_df = pd.merge(df_feature4, df_rct, on='PDF File', how='inner')
cleaned_df = merged_df[~(merged_df['Number of Patients'].isna() | merged_df['Number of RCTs'].isna())]

print(cleaned_df)
# cleaned_df.to_csv("cleaned_df.csv", index = False)

source_folder = 'downloaded_committee_papers'
exclude_folder = '50_Committee_papers'
destination_folder = '200_committee_papers'
"""
# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get set of filenames to exclude (those already in "50_Committee_papers")
excluded_files = set(os.listdir(exclude_folder))

# Filter out excluded files from the DataFrame
filtered_df = cleaned_df[~cleaned_df['PDF File'].isin(excluded_files)]

# Define how many PDFs per category
required_counts = {
    'Approved': 51,
    'Rejected': 26,
    'Conditional': 83
}

# Function to copy PDFs
def copy_category_pdfs(category, count):
    subset = filtered_df[filtered_df['Decision'] == category]
    available_count = len(subset)

    if available_count < count:
        print(f"⚠️ Not enough '{category}' PDFs. Needed: {count}, Available: {available_count}")
        count = available_count  # Adjust to copy only available

    for filename in subset['PDF File'].head(count):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"Copied ({category}): {filename}")
        else:
            print(f"File not found in source: {filename}")

# Process each category
for category, count in required_counts.items():
    copy_category_pdfs(category, count)
"""
    
# Path to the folder where files were copied
copied_folder = '200_committee_papers'

# Get list of copied PDF filenames
copied_filenames = set(os.listdir(copied_folder))
print(len(copied_filenames))

# Filter df_feature4 for rows where 'PDF File' matches one of the copied files
copied_metadata_df = df_feature4[df_feature4['PDF File'].isin(copied_filenames)][
    ['PDF File', 'TA Code', 'Decision']
].reset_index(drop=True)

# Get the set of filenames from df_feature4
feature4_filenames = set(df_feature4['PDF File'])

# Find unmatched filenames
unmatched_filenames = copied_filenames - feature4_filenames

print(f" {len(unmatched_filenames)} PDF files in the folder have no match in df_feature4:")
for f in sorted(unmatched_filenames):
    print(f)

# Show result
print("DataFrame with copied file metadata:")
print(copied_metadata_df)
copied_metadata_df.to_csv('200_committee_papers_info.csv', index=False)
