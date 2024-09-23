import ast, os, time
import pandas as pd
from typing import List, Dict

from create_dataset.data_extractor import LoadGoDollarstreetData, DataProcessor
# from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
# from models.llavamodel.llava.llava.eval.run_llava import eval_model
from create_dataset.dataset_processor import DatasetCreator


def main() -> None:
    """
    Main function to load data, process WVS questions and options, filter common countries, 
    and create the final dataset with image paths and prompts.

    Returns:
        None
    """
    # Define the data folder path
    data_folder: str = "data"

    data_loader = LoadGoDollarstreetData("Anthropic/llm_global_opinions", f"{data_folder}/dollarstreet/images_v2.csv")
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
    data_processor = DataProcessor(selections)

    # Load the DollarStreet data
    dollarstreet_data = data_loader.get_dollarstreet_data()

    # Get the list of common countries between WVS data and DollarStreet data
    common_countries: List[str] = data_processor.filter_common_countries(dollarstreet_data['country.name'].unique())

    # Prepare family data with photo paths and income information for the common countries
    family_photo_income_data = data_processor.prepare_family_data(dollarstreet_data, common_countries)

    # Create a copy of the family data
    family_data = family_photo_income_data.copy()

    # Add full image paths by concatenating folder path with relative image path
    family_data.loc[:, 'imageFullPath'] = f"{data_folder}/dollarstreet/" + family_data['imageRelPath']

    # Get the number of questions to process
    def_n_questions: int = len(questions)

    # Initialize the dataset processor with image paths, income, questions, selections, and options
    processor = DatasetCreator(
        family_data['imageFullPath'].tolist(),            # Image paths
        dollarstreet_data['income'],                      # Income data
        questions[:def_n_questions],                      # WVS questions
        selections[:def_n_questions],                     # WVS selections
        labeled_options                                   # Labeled options for the questions
    )

    # Create the dataset by processing family data and associating images with prompts and options
    all_data: List[Dict[str, any]] = processor.create_data(family_data)

    # Save the list of dictionaries to a DataFrame
    df: pd.DataFrame = pd.DataFrame(all_data)

    # Sort the DataFrame by country and then income
    df = df.sort_values(by=['country', 'income'], ascending=[True, True], ignore_index=True)

    # Output the number of samples in the final DataFrame
    print(f"Number of samples in dataframe: {df.shape[0]}")

    csv_file_name = f"{data_folder}/llava_data.csv"

    # delete file if it exists
    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)
    df.to_csv(csv_file_name, index=False)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time} seconds")
