# %%
import pandas as pd

# %%
import sys
sys.path.append('/home/srishti/dev/cultural_values')

# %% [markdown]
# ### Import LLM global opinion data from hugging face

# %%
from datasets import load_dataset

# Hugging face dataset
go_dataset = "Anthropic/llm_global_opinions"
go_dataset = load_dataset(go_dataset)
print(f"LLM GO Dataset Details: {go_dataset}")

# load first 5 rows of the dataset from the train split of huggimngface dataset
train_data = go_dataset['train']

# choose all data where "source" is "WVS". Data is huggingface dataset
train_data_wvs = train_data.filter(lambda x: x['source'] == 'WVS')
print(f"Train Data WVS Details: {train_data_wvs}")

# Select questions, selections and options
train_wvs_questions = train_data_wvs['question']
train_wvs_selections = train_data_wvs['selections']
train_wvs_options = train_data_wvs['options']

# %% [markdown]
# Print train and test data for WVS-7

# %%
import random, ast
from collections import defaultdict

# choose a random number between 1 and 2556
max_num = 353
choose_num = random.randint(1, max_num)
print(f"Random number between 1 and {max_num}: {choose_num}")

# Select vaue from the list based on the random number
print("Question:")
print(train_wvs_questions[choose_num:choose_num+1])
print("---"*50)

print("Countries and Responses:")
country_responses = train_wvs_selections[choose_num:choose_num+1]
responses_dict = ast.literal_eval(country_responses[0].strip("defaultdict(<class 'list'>, ").strip(")"))
print(responses_dict)
print("Countries:")
print(responses_dict.keys())
print("---"*50)

print("Options:")
print(train_wvs_options[choose_num:choose_num+1])

# %% [markdown]
# Find all the countries in the WVS dataset

# %%
wvs_countries = []
for response in train_wvs_selections:
    each_response = ast.literal_eval(response.strip("defaultdict(<class 'list'>, ").strip(")"))
    each_keys = each_response.keys()
    wvs_countries.extend(each_keys)
wvs_countries = list(set(wvs_countries))
print(f"Unique Keys: {wvs_countries}")    
print(f"Total Unique Keys: {len(wvs_countries)}")

# %% [markdown]
# -------------------------------------------------------------------------------------------------------------

# %% [markdown]
# #### Load Dollarstreet Images

# %%
import pandas as pd

dollarstreet_data = pd.read_csv('data/dollarstreet/images_v2.csv')
print(f"Number of rows in the dollarstreet dataset: {dollarstreet_data.shape[0]}")
dollarstreet_data.head()

# %% [markdown]
# Find all country names in the dollarstreet_data

# %%
dollarstreet_countries = dollarstreet_data['country.name'].unique()
print(f"Number of unique countries in the dollarstreet dataset: {len(dollarstreet_countries)}")

# Print the unique countries in the dollarstreet dataset
print(dollarstreet_countries)

# %% [markdown]
# #### Find the common countries between the two: wvs_countries and dollarstreet_countries dataset
# 

# %%
# Find the common countries between the two wvs_countries and dollarstreet_countries
common_countries = list(set(wvs_countries).intersection(dollarstreet_countries))
print(f"Common countries between the two datasets: {common_countries}")
print(f"Number of common countries between the two datasets: {len(common_countries)}")

# %%
# In the "topics" column of dollarstreet_data, select rows with topics= Family and Family snapshots and with country.name in the common countries list
family_data = dollarstreet_data[dollarstreet_data['topics'].isin(['Family', 'Family snapshots']) & dollarstreet_data['country.name'].isin(common_countries) & dollarstreet_data['type'].isin(['image'])]
print(f"Number of rows in the family_data dataset: {family_data.shape[0]}")

# %%
family_data.head()

# %% [markdown]
# #### Analysis using VLM Model
#   TO DO:
# - For each common country, use the corresponding image from the dollarstreet dataset and pass WVS questions
#     - (1) to the VLM model (LLM decoder) to get answers
#     - (2) to the VLM model (both image-text decoder) to get answers
# 
# - Compare the answers from the two models and see which one is better
#     - Define a metric to compare the answers
#     - Compare the answers for each question across all countries
# 
# - Run this for multiple seeds and see if the results are consistent
#     - Get the average score for each question across all countries
#     - Get distribution shift for each question across all countries
# 
# 

# %% [markdown]
# 

# %% [markdown]
# Create full image path in the dollarstreet dataset

# %%
# add prefix to the imageRelPath column. Dir: ../data/dollarstreet/
# create a new column with the full path of the image and name it as "imageFullPath"

family_data['imageFullPath'] = "data/dollarstreet/" + family_data['imageRelPath'].copy()
family_data.head()

# %% [markdown]
# ##### Load multimodal input
# 
# * Image: Dollarstreet image
# * Text: WVS questions with prompt

# %%
from runllava_hf import LLAVAProcessor

len_image_data = family_data.shape[0]

# convery pandas series to list
image_files = family_data['imageFullPath'].tolist()

# This function will convert the user input into a boolean
def parse_user_input(input_value):
    # Define the positive responses that count as True
    affirmative_responses = {"YES", "Yes", "yes", "Y", "y"}
    # Define the negative responses that count as False
    negative_responses = {"NO", "No", "no", "N", "n"}
    
    # Assert that the input is valid
    assert input_value in affirmative_responses.union(negative_responses), "Invalid choice. Please choose Yes or No."
    
    # Return True if the user input is an affirmative response, otherwise False
    return input_value in affirmative_responses

print(f"Total number of images in the family_data dataset: {len_image_data}")

def main(user_input=False):
    """
    The main function to run the LLAVA processing, either based on user input or automatically.

    Parameters:
        user_input (bool): If True, the function will prompt for user input. If False, it runs automatically.
    """

    processor = LLAVAProcessor(img_files_paths=image_files,
                               wvs_questions=train_wvs_questions,
                               wvs_options=train_wvs_options)

    user_input = input("Do you want to use images? (Yes/No): ")
    use_images = parse_user_input(user_input)

    if user_input:
        # default is all: total_images
        num_samples = int(input("How many samples do you want to process? (Default is all): ") or len_image_data) 

        data = processor.process_data(family_data, num_samples, use_images)
        file_name = f'llava_output_img_{use_images}.csv'
        processor.save_results(data, file_name)
    else:
        num_samples = len_image_data
        print(f"Processing {num_samples} samples...")
        print("--------------------------------------------------")

        for use_images in [True, False]:
            data = processor.process_data(family_data, num_samples, use_images)
            file_name = f'llava_output_img_{use_images}.csv'
            processor.save_results(data, file_name)

if __name__ == "__main__":
    main(user_input=True)  # Modify as needed for automated or manual execution





