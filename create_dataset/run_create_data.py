import ast, os, time
import pandas as pd
import shutil
from typing import List, Dict
from pathlib import Path

from data_extractor import LoadGoDollarstreetCVQAData, DataExtractor
# from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
# from models.llavamodel.llava.llava.eval.run_llava import eval_model
from dataset_processor import DatasetCreator


def main() -> None:
    """
    Main function to load data, process WVS questions and options, filter common countries, 
    and create the final dataset with image paths and prompts.

    Returns:
        None
    """
    # Define the data folder path
    
    # CHECK THIS PATH
    ds_data_folder: str = "/projects/belongielab/people/vsl333/ds"
    root_dir = "/home/vsl333/cultural_values"
    cvqa_data_folder: str = "cvqa_chosen"
    created_csv_data_dir: str = Path(f"{root_dir}/created_csv_data")
    if os.path.exists(created_csv_data_dir):
        shutil.rmtree(created_csv_data_dir)
    os.makedirs(created_csv_data_dir)


    data_loader = LoadGoDollarstreetCVQAData("Anthropic/llm_global_opinions", f"{ds_data_folder}/dollarstreet/images_v2.csv", f"{cvqa_data_folder}/metadata.csv")
    
    # Get WVS questions, selections, and options
    questions: List[str]
    selections: List[str]
    unlabelled_options: List[str]
    questions, selections, unlabelled_options = data_loader.get_wvs_data()

    # Convert string representations of options to Python lists
    unlabelled_options: List[List[str]] = [ast.literal_eval(each_sublabel.strip()) for each_sublabel in unlabelled_options]

    # Add letter labels to each option in the sublist (e.g., (A) Option 1, (B) Option 2, etc.)
    letter_labeled_options: List[List[str]] = [
        [f"({chr(65 + i)}) {option}" for i, option in enumerate(sublist)] for sublist in unlabelled_options
    ]

    # Join the options with labels into a single string, removing unnecessary quotes
    labeled_options: List[str] = [", ".join(sublist) for sublist in letter_labeled_options]

    # Initialize the data processor for country selections
    data_extractor = DataExtractor(selections)

    # Load the DollarStreet data
    dollarstreet_data = data_loader.get_dollarstreet_data()
    cvqa_data = data_loader.get_cvqa_data()


    # Get the list of common countries between WVS data and DollarStreet data
    common_countries: List[str] = data_extractor.filter_common_countries(dollarstreet_data['country.name'].unique(), cvqa_data['country'].unique())

    # Prepare family data with photo paths for the common countries.
    # Add full image paths by concatenating folder path with relative image path
    ds_family_data = data_extractor.prepare_ds_family_data(dollarstreet_data, common_countries)
    ds_country_data = ds_family_data.copy()
    ds_country_data.loc[:, 'imageFullPath'] = f"{ds_data_folder}/dollarstreet/" + ds_country_data['imageRelPath']
    ds_country_data.loc[:, 'image_code'] = "ds"
    ds_country_data.sort_values(by=['country.name'], ascending=True, ignore_index=True, inplace=True)

    cvqa_country_data = data_extractor.prepare_cvqs_img_data(cvqa_data, common_countries)
    cvqa_country_data.sort_values(by=['country'], ascending=True, ignore_index=True, inplace=True)
    cvqa_country_data.loc[:, 'image_code'] = "cvqa"

    # Get the number of questions to process
    def_n_questions: int = len(questions)

    # Initialize the dataset processor with image paths, income, questions, selections, and options
    # TO DO: Add income later
    processor = DatasetCreator(
        ds_country_data['imageFullPath'].tolist(),            # Image paths
        questions[:def_n_questions],                      # WVS questions
        selections[:def_n_questions],                     # WVS selections
        labeled_options                                   # Labeled options for the questions
    )


    # Create country specfic dataframe for cvqa data
    cvqa_percountry_list = []
    for unique_country in cvqa_country_data['country'].unique():
        cvqa_u_country_data = cvqa_country_data[cvqa_country_data['country'] == unique_country]
        cvqa_percountry_list.append(cvqa_u_country_data)

    
    # Create the dataset by processing family data and associating images with prompts and options
    country_data_ds_cvqa= [ds_country_data] + cvqa_percountry_list

    for each_country_data in country_data_ds_cvqa:
        all_data: List[Dict[str, any]] = processor.create_data(each_country_data)
        # breakpoint()

        # Save the list of dictionaries to a DataFrame
        df: pd.DataFrame = pd.DataFrame(all_data)

        # Sort the DataFrame by country
        df = df.sort_values(by=['country'], ascending=[True], ignore_index=True)

        
        if each_country_data["image_code"].unique() == "ds":
            print(f"Number of samples in dataframe: {df.shape[0]}")
            df["image_code"] = "ds"

            # breakpoint()      
            # find "income" column in ds_country_data and add it to df as "income". Match by "id"
            df = pd.merge(df, ds_country_data[['id', 'income']], on='id', how='left')
            csv_file_name = f"{created_csv_data_dir}/ds_people.csv"
            print(f"Saving to {csv_file_name}")
            if os.path.exists(csv_file_name):
                os.remove(csv_file_name)
            df.to_csv(csv_file_name, index=False)
            print(f"Saved to {csv_file_name}")

        elif each_country_data["image_code"].unique() == "cvqa":
            print(f"Number of samples in dataframe: {df.shape[0]}")
            df["image_code"] = "cvqa"
            country_name = df['country'].unique()[0]   
            csv_file_name = f"{created_csv_data_dir}/cvqa_{country_name}.csv"
            print(f"Saving to {csv_file_name}")
            if os.path.exists(csv_file_name):
                os.remove(csv_file_name)
            df.to_csv(csv_file_name, index=False)
            print(f"Saved to {csv_file_name}")
        else:
            raise ValueError("Image code not recognized. Please check the image_code column in the dataframe.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time} seconds")
