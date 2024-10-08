import ast, os, time
import pandas as pd
import shutil
from typing import List, Dict
from pathlib import Path
import uuid

from data_extractor import LoadGoDollarstreetCVQAData, DataExtractor
# from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
# from models.llavamodel.llava.llava.eval.run_llava import eval_model
from dataset_processor import DatasetCreator
import random


def main() -> None:
    """
    Main function to load data, process WVS questions and options, filter common countries, 
    and create the final dataset with image paths and prompts.

    Returns:
        None
    """
    # Define the data folder path
    
    # CHECK THIS PATH
    cvqa_data_folder: str = Path("/home/vsl333/cultural_values/datasets/cvqa_images")

    all_data = []
    country_folder_paths = [f for f in cvqa_data_folder.iterdir() if f.is_dir()]
    category_folders = [f for country_folder in country_folder_paths for f in country_folder.iterdir() if f.is_dir()]

    for category_folder in category_folders:
        image_data = []
        url_txt_file = [ category_folder / "url.txt", category_folder / "urls.txt"]
        # if category_folder / "urls.txt" exists rename it to url.txt
        if url_txt_file[1].exists():
            shutil.move(url_txt_file[1], url_txt_file[0])

        # print(f"Reading {url_txt_file}")
        url_txt_path = category_folder / "url.txt"
        if url_txt_path.exists():
            with open(url_txt_path, 'r') as file:
                for line in file:
                    image_name, image_url = line.split(',')
                    image_path = category_folder / image_name
                    image_path_base = os.path.join(*image_path.parts[-3:-1], image_name)
                    if image_path.exists() and image_path.suffix in ['.jpg', '.jpeg', '.png']:
                        image_data.append({
                            'image_name': image_name,
                            'image_url': image_url,
                            'image_path': str(image_path_base),
                            'category': category_folder.name,
                            'country': category_folder.parent.name
                        })
        all_data.extend(image_data)
    
    df = pd.DataFrame(all_data)
    df = df[df['category'] != 'plants_and_animals']
    # exclude Iran
    df = df[df['country'] != 'Iran']
    print(df['country'].value_counts())
    # Set a seed value for reproducibility
    seed_value = 42
    random.seed(seed_value)

    df['id'] = [str(uuid.UUID(int=random.getrandbits(128))) for _ in range(len(df))]
    df.to_csv(cvqa_data_folder / "metadata.csv", index=False)

    # # read the csv file
    df = pd.read_csv(cvqa_data_folder / "metadata.csv")

    # cvqa_data_folder: str = "cvqa_chosen"

    created_csv_data_csv: str = cvqa_data_folder / "cvqa_wvs_metadata.csv"
    if os.path.exists(created_csv_data_csv):
        os.remove(created_csv_data_csv)

    # True if 'country.name' also has ['United States', Switzerland', 'Denmark'] only for DollarStreet data
    # ds_expansion = True

    data_loader = LoadGoDollarstreetCVQAData("Anthropic/llm_global_opinions", cvqa_data_folder, dollarstreet_csv_path = None, cvqa_csv_path = f"{cvqa_data_folder}/metadata.csv")
    
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
    # dollarstreet_data = data_loader.get_dollarstreet_data()
    cvqa_data = data_loader.get_cvqa_data()


    # Get the list of common countries between WVS data and DollarStreet data
    common_countries_lst: List[str] = data_extractor.filter_common_countries(dollarstreet_countries = None, cvqa_countries = cvqa_data['country'].unique())

    # if ds_expansion:
    #     common_countries = common_countries_lst + ['United States', 'Switzerland', 'Denmark']
    # else:
    common_countries = common_countries_lst

    # Prepare family data with photo paths for the common countries.
    # Add full image paths by concatenating folder path with relative image path
    # ds_family_data = data_extractor.prepare_ds_family_data(cvqa_data, common_countries)
    # ds_country_data = ds_family_data.copy()


    # ds_country_data.loc[:, 'imageFullPath'] = f"{ds_data_folder}/dollarstreet/" + ds_country_data['imageRelPath']
    # create new column 'image_full_path' and add it to the dataframe
    # ds_country_data.loc[:, 'image_full_path'] = f"{ds_data_folder}/" + ds_country_data['image_path']
    # ds_country_data.loc[:, 'image_code'] = "ds"
    # ds_country_data.sort_values(by=['country.name'], ascending=True, ignore_index=True, inplace=True)    

    cvqa_country_data = data_extractor.prepare_cvqs_img_data(cvqa_data, common_countries)
    cvqa_country_data.sort_values(by=['country'], ascending=True, ignore_index=True, inplace=True)
    cvqa_country_data.loc[:, 'image_code'] = "cvqa"
    cvqa_country_data.loc[:, 'image_full_path'] = f"{cvqa_data_folder}/" + cvqa_country_data['image_path']

    # Print image_full_path and image_code

    # Get the number of questions to process
    def_n_questions: int = len(questions)


    # Initialize the dataset processor with image paths, income, questions, selections, and options
    # TO DO: Add income later
    processor = DatasetCreator(
        cvqa_country_data['image_full_path'].tolist(),            # Image paths
        questions[:def_n_questions],                      # WVS questions
        selections[:def_n_questions],                     # WVS selections
        labeled_options                                   # Labeled options for the questions
    )


    # to do: do you want to create a dataset for each country?
    
    # # Create country specfic dataframe for cvqa data
    # cvqa_percountry_list = []
    # for unique_country in cvqa_country_data['country'].unique():
    #     cvqa_u_country_data = cvqa_country_data[cvqa_country_data['country'] == unique_country]
    #     cvqa_percountry_list.append(cvqa_u_country_data)

    
    # Create the dataset by processing family data and associating images with prompts and options
    country_data_ds_cvqa= [cvqa_country_data]

    total_df_list = []
    for each_country_data in country_data_ds_cvqa:
        all_data: List[Dict[str, any]] = processor.create_data(each_country_data)

        # breakpoint()

        # Save the list of dictionaries to a DataFrame
        df: pd.DataFrame = pd.DataFrame(all_data)

        # Sort the DataFrame by country
        df = df.sort_values(by=['country'], ascending=[True], ignore_index=True)

        print(f"Number of samples in dataframe: {df.shape[0]}")
        df["image_code"] = "cvqa"

        total_df_list.append(df)

    total_df = pd.concat(total_df_list, ignore_index=True)
    total_df.to_csv(created_csv_data_csv, index=False, header=True, sep=',', encoding='utf-8')
    print(f"Saved to {created_csv_data_csv}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time} seconds")
