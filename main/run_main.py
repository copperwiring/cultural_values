import os, shutil, time, logging, ast, sys
import pandas as pd
# from tqdm import tqdm as tdqm
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.disable(logging.CRITICAL)  # Disables all logging calls of severity 'CRITICAL' and below
from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
from models.llavamodel.llava.llava.eval.run_llava import eval_model


logging.info("Creating output directory")


class Go_WVS_Img_Dataset(Dataset):
    def __init__(self, data):
        self.img_id = data['id']
        self.image_path = data['image_path']
        self.country = data['country']
        self.image_code = data['image_code']
        self.income = data['income'] if 'income' in data.columns else "N"
        self.question_text = data['question_text']
        self.country_prompt = data['country_prompt']
        self.generic_prompt = data['generic_prompt']
        self.option_labels = data['option_labels']
        self.full_options = data['full_options']
        self.selection_answers = data['selection_answers']


    def __len__(self):
        return len(self.img_id) # number of samples

    def __getitem__(self, idx):
        return {
            'img_id': self.img_id[idx],
            'image_path': self.image_path[idx],
            'country': self.country[idx],
            'image_code': self.image_code[idx],
            'income': self.income[idx] if hasattr(self, 'income') and idx < len(self.income) else "N",
            'question_text': self.question_text[idx],
            'country_prompt': self.country_prompt[idx],
            'generic_prompt': self.generic_prompt[idx],
            'option_labels': self.option_labels[idx],
            'full_options': self.full_options[idx],
            'selection_answers': self.selection_answers[idx]
        }
        
class DatasetManager:
    def __init__(self, data, batch_size=2,num_workers=1, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_dataloader(self):
        eval_dataset = Go_WVS_Img_Dataset(self.data)
        batch_data = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return batch_data

class ModelEvaluator:
    def __init__(self, model_path, dataloader):
        self.model_path = model_path
        self.dataloader = dataloader
        self.results_dict = {}
        self.combined_results = {}
        self.model_path = model_path

    def evaluate_model(self, prompts_batch, img_files_batch, letter_options, full_options):
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompts_batch,
            "conv_mode": None,
            "image_file": img_files_batch,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
            })()
    
        batch_results = eval_model(args, prompts_batch, img_files_batch, letter_options, full_options)
        return batch_results

    def evaluate_batches(self):
        # Loop through the dataset and evaluate each batch
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            prompts_batch = batch['generic_prompt']
            img_files_batch = batch['image_path']
            letter_options = batch['option_labels']
            full_options = batch['full_options']
            
            # convert selection_answers to list of floats
            batch['selection_answers'] = [ast.literal_eval(each_sublabel.strip()) for each_sublabel in batch['selection_answers']]
            
            # Pass the batched data to the evaluation function
            batch_results = self.evaluate_model(prompts_batch, img_files_batch, letter_options, full_options)
            
            # Store the results of batched values and batch results
            batch_dict = {**batch, **batch_results}
            
            # Store the results of batch_dict in results_dict
            for key, value in batch_dict.items():
                if key not in self.results_dict:
                    self.results_dict[key] = []
                self.results_dict[key].append(value)

        self.combine_results()

    def combine_results(self):
        # Combine the results of all batches
        for key, value in tqdm(self.results_dict.items()):
            self.combined_results[key] = [item for sublist in value for item in sublist]

    def save_results(self, output_dir, csv_file_name):
        # Save the results to a csv file
        combined_results_df = pd.DataFrame(self.combined_results)

        # Put 'selection_answers' in the right side of 'prob_percent_values'
        selection_answers = combined_results_df['selection_answers']
        prob_percent_values = combined_results_df['prob_percent_values']
        combined_results_df.drop(columns=['selection_answers', 'prob_percent_values'], inplace=True)
        combined_results_df['prob_percent_values'] = prob_percent_values
        combined_results_df['selection_answers'] = selection_answers

        print(f"Length of results_df: {len(combined_results_df)}")
        output_file = os.path.join(output_dir, f"{csv_file_name.split('.')[0]}_results.csv")
        
        # Delete file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            
        combined_results_df.to_csv(output_file, index=False)

def main(csv_file_path, model_path, output_dir, batch_size, num_workers):
    start_time = time.time()

    data = pd.read_csv(csv_file_path)
    # data = data[:4]  # select last n rows for testing
    data = data.sort_values(by=['country'], ascending=[True], ignore_index=True)
    # Initialize Dataset Manager
    dataset_manager = DatasetManager(data, batch_size=batch_size, num_workers=num_workers)
    dataloader = dataset_manager.get_dataloader()

    # Initialize Model Evaluator
    evaluator = ModelEvaluator(model_path, dataloader)

    # Evaluate batches and save results
    evaluator.evaluate_batches()
    evaluator.save_results(output_dir, csv_file_path.split('/')[-1])

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":

    # add args parameter to receive csv file path, model path, output directory
    parser = argparse.ArgumentParser(description='Evaluate VLM model on WVS dataset')
    parser.add_argument('--csv_file_path', type=str, help='Path to the csv file containing the dataset')
    parser.add_argument('--model_name', type=str, help='Name of the model to be evaluated')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')

    # e.g. 
    
    args = parser.parse_args()
    model_path = args.model_name
    output_dir = args.output_dir
    csv_file_path = args.csv_file_path


    # Call the main function
    main(csv_file_path, model_path, output_dir, args.batch_size, args.num_workers)


# Loop through the dataset and evaluate each batch
# results_dict = {}; combined_results = {}
# for i, batch in tdqm(enumerate(dataset_loader), total=len(dataset_loader)):
#     prompts_batch, img_files_batch, letter_options, full_options = batch['generic_prompt'], batch['image_path'], batch['option_labels'], batch['full_options']

#     # convert selection_answers to list of floats
#     batch['selection_answers'] = [ast.literal_eval(each_sublabel.strip()) for each_sublabel in batch['selection_answers']]
#     # Pass the batched data to the evaluation function
#     batch_results = evaluate_model(prompts_batch, img_files_batch, letter_options, full_options)

#     # Store the results of batche values and batch results
#     batch_dict = {**batch, **batch_results}

#     # Store the results of batch_dict in results_dict
#     for key, value in batch_dict.items():
#         if key not in results_dict:
#             results_dict[key] = []
#         results_dict[key].append(value)

# # Combine the results of all batches
# from tqdm import tqdm
# for key, value in tqdm(results_dict.items()):
#     combined_results[key] = [item for sublist in value for item in sublist]


# # Save the results to a csv file
# combined_results_df = pd.DataFrame(combined_results)

# # Put 'selection_answers' in the right side of 'prob_percent_values'
# selection_answers = combined_results_df['selection_answers']
# prob_percent_values = combined_results_df['prob_percent_values']
# # move selection_answers to the right side of prob_percent_values
# combined_results_df.drop(columns=['selection_answers', 'prob_percent_values'], inplace=True)
# combined_results_df['prob_percent_values'] = prob_percent_values
# combined_results_df['selection_answers'] = selection_answers


# print(f"length of results_df: {len(combined_results_df)}")
# output_file = os.path.join(output_dir, "results.csv")
# # delete file if it exists
# if os.path.exists(output_file):
#     os.remove(output_file)
# combined_results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")




