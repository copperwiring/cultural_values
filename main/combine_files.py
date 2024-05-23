# Import combined_output.csv and split it into three separate files so that file size is small

import os
import pandas as pd
import shutil

# Read all files in the output directory and set column "set_index" as index for each file
output_dir = "output/"
output_file_name = "combined.csv"

# List all files in the directory
files_in_directory = os.listdir(output_dir)

# Filter files that start with 'a' and end with '.csv'
files_to_delete = [file for file in files_in_directory if file.startswith('combined') and file.endswith('.csv')]

# Delete the filtered files
for file in files_to_delete:
    os.remove(os.path.join(output_dir, file))
    print(f"Deleted: {file}")

csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

dataframes = [pd.read_csv(os.path.join(output_dir, f), index_col="set_index") for f in csv_files]

# Concatenate all dataframes along the columns
combined_data = pd.concat(dataframes, axis=1)
combined_data.to_csv(os.path.join(output_dir, output_file_name))

data = combined_data
column_name = 'llava_output?_use_images_False_use_country_name_True'

# check if "llava_output?_use_images_False_use_country_name_True" column exists
if column_name not in data.columns:
    print(f"Column {column_name} does not exist in the data")
    exit(1)

# Function to ensure all values are in single quotes
def ensure_single_quotes(value):
    # Remove any existing quotes (both single and double)
    cleaned_value = value.replace('"', '').replace("'", "")
    # Enclose in single quotes
    return cleaned_value

# value of this column: 'llava_output?_use_images_False_use_country_name_True' is a string
# Remove all single and double quotes and make them standardized
data[column_name]= data[column_name].apply(ensure_single_quotes)

# get first row value only
first_row_value = data[column_name].iloc[0]
print(f"First row value: {first_row_value}")


# Define the allowed responses
allowed_responses = [
    'Completely dissatisfied', 'Rather dissatisfied', 'Rather satisfied',
    'Completely satisfied', "Don't know", 'No answer'
]

# Filter the dataframe for one specific llava_output column
filtered_data = data[data[column_name].isin(allowed_responses)]
print(f"Number of rows in the original data: {data.shape[0]}")
print(f"Number of rows in the filtered data: {filtered_data.shape[0]}")

# Save the filtered data to a new CSV file
filtered_data.to_csv(os.path.join(output_dir, "combined_output_filtered.csv"))


