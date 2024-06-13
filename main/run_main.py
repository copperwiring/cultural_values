from data_handling import DataLoader, DataProcessor
from llava_processor import LLAVAProcessor
import os, shutil
import pandas as pd
import time, ast

import logging
logging.disable(logging.CRITICAL)  # Disables all logging calls of severity 'CRITICAL' and below



def main(user_choice=False):
    data_loader = DataLoader("Anthropic/llm_global_opinions", "data/dollarstreet/images_v2.csv")
    questions, selections, unlabelled_options = data_loader.get_wvs_data()
    unlabelled_options = [ast.literal_eval(each_sublabel.strip()) for each_sublabel in unlabelled_options]
    # Process each sublist to add labels
    letter_labeled_options = [
        [f"({chr(65 + i)}) {option}" for i, option in enumerate(sublist)] for sublist in unlabelled_options
        ]
     # remove quotes from the options
    labeled_options = [", ".join(sublist) for sublist in letter_labeled_options]
        
    dollarstreet_data = data_loader.get_dollarstreet_data()

    print(f"Number of samples in original dollarstreet data: {dollarstreet_data.shape[0]}")
    print(f"Number of question in original WVS data: {len(questions)}")
    print(f"Number of options in original WVS data: {len(labeled_options)}")

    print(f"If we were to run the model on all samples, we would have {dollarstreet_data.shape[0] * len(questions)} samples to test.")

    data_processor = DataProcessor(selections)
    common_countries = data_processor.filter_common_countries(dollarstreet_data['country.name'].unique())

    family_photo_income_data = data_processor.prepare_family_data(dollarstreet_data, common_countries)
    family_data = family_photo_income_data.copy()
    family_data.loc[:, 'imageFullPath'] = "data/dollarstreet/" + family_data['imageRelPath']

    # Number of samples in filtered dollarstreet data in each common country
    count_country_samples = family_data[family_data['country.name'].isin(common_countries)].groupby('country.name').size()
    print(f"Filtered data each: \n{count_country_samples}")
    print("*"*60)


    print("*"*60)
    print(f"Number of common countries in both datasets: {len(common_countries)}")
    print(f"Total samples filtered dollarstreet data: {family_data.shape[0]} * {len(questions)} = {family_data.shape[0] * len(questions)}")
    print("*"*60)

    num_samples = int(input("How many samples do you want to process? (Default is all): ") or family_data.shape[0])
    def_n_questions = len(questions) # Number of questions to process
    # def_n_questions = 7 # Number of questions to process

    print(f"In default model -  We will process {num_samples} samples with {def_n_questions} questions.")
    print(f"Total data points to process: {num_samples * def_n_questions} in different combinations.")

    output_dir = "output/"
    # Delete directory if it exists and create a new one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if user_choice:
        use_images = input("Do you want to use images? (Yes/No): ").strip().lower() in ['yes', 'y'] # Default is Yes
        use_country_name = input("Do you want to use country name in the prompt? (Yes/No): ").strip().lower() in ['no', 'n'] # Default is No

        processor = LLAVAProcessor(family_data['imageFullPath'].tolist(), questions[:def_n_questions], selections[:def_n_questions], labeled_options)
        processed_data = processor.process_data(family_data, num_samples, use_images, use_country_name)

        file_name = f'llava_output_img_{use_images}_country_{use_country_name}.csv'
        file_path = os.path.join(output_dir, file_name)
        processor.save_results(processed_data, file_path)
        print(f"Processing complete. Results saved to {file_path}.")
    else:
        for use_images in [True, False]:
            for use_country_name in [False]: # add True if needed
                print("*"*60)
                print(f"Combination: Images - {use_images}, Country Name - {use_country_name}")
                processor = LLAVAProcessor(family_data['imageFullPath'].tolist(), questions[:def_n_questions], selections[:def_n_questions], labeled_options)
                processed_data = processor.process_data(family_data, num_samples, use_images, use_country_name)
                
                file_name = f'llava_img_{use_images}_country_{use_country_name}.csv'
                file_path = os.path.join(output_dir, file_name)
                processor.save_results(processed_data, file_path, use_images, use_country_name)
                print(f"Processing complete. Results saved to {file_path}.")

if __name__ == "__main__":
    # count time it takes to process all combinations
    start_time = time.time()
    main(user_choice=False)  # Modify as needed for automated (False) or manual (True) execution
    print(f"Total time taken: {time.time() - start_time} seconds.")






