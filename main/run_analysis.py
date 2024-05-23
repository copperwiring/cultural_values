import os
import pandas as pd

base_dir = "output_sample_1/"
base_file = "combined_output_filtered.csv"

# Read the combined file
data = pd.read_csv(os.path.join(base_dir, base_file))

column_image = "img_path?_use_images_True_use_country_name_True"
column_question = "question?_use_images_False_use_country_name_False"

# ic stands for "image and country", i stands for "image", noic stands for "no image and country" 
column_ic = 'llava_output?_use_images_True_use_country_name_True'
column_i = 'llava_output?_use_images_True_use_country_name_False'
column_noic = 'llava_output?_use_images_False_use_country_name_True'

# Function to ensure all values are in single quotes
def ensure_single_quotes(value):
    # Remove any existing quotes (both single and double)
    cleaned_value = value.replace('"', '').replace("'", "")
    # Enclose in single quotes
    return cleaned_value

# value of this column: 'llava_output?_use_images_False_use_country_name_True' is a string
# Remove all single and double quotes and make them standardized
data[column_ic] = data[column_ic].apply(ensure_single_quotes)
data[column_i] = data[column_i].apply(ensure_single_quotes)
data[column_noic] = data[column_noic].apply(ensure_single_quotes)

# get first row value only
print(data[column_ic].iloc[0])
print(data[column_i].iloc[0])
print(data[column_noic].iloc[0])


country_column = "country?_use_images_True_use_country_name_True"
# Get this column but rename it to country
country_data = data[[country_column]]
country_data.columns = ["country"]  


# Filter out these columns: column_image, column_question, column_ic, column_i 
filtered_data = data[[column_image, country_column, column_question, column_ic, column_i, column_noic]]

# Save the filtered data
filtered_file = "analysis_filtered.csv"
filtered_data.to_csv(os.path.join(base_dir, filtered_file), index=False)
print(f"Filtered data saved to {os.path.join(base_dir, filtered_file)}")

# ANALYSIS 1
# All column_question where column_ic is not equal to column_i
analysis1 = filtered_data[filtered_data[column_ic] != filtered_data[column_i]][column_question]
analysis2 = filtered_data[filtered_data[column_ic] != filtered_data[column_noic]][column_question]

print(f"Analysis 1: When Use Image But Uee of Country Name Is Switched")
print(f"% of samples changed: {analysis1.shape[0] / filtered_data.shape[0] * 100:.2f}%")

print(f"Analysis 2: When Use Country But Use of Image  Is Switched")
print(f"% of samples changed: {analysis2.shape[0] / filtered_data.shape[0] * 100:.2f}%")


# ANALYSIS 2# Define the allowed responses
# allowed_responses = [
#     'Completely dissatisfied', 'Rather dissatisfied', 'Rather satisfied',
#     'Completely satisfied', "Don't know", 'No answer'
# ]

# TO DO
# 1. Make all *dissatisfied* responses as 'dissatisfied'
# 2. Make all *satisfied* responses as 'satisfied'
# 3. Make all *don't know* responses as 'don't know'
# 4. Make all *no answer* responses as 'no answer'






