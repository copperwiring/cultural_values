import os, shutil, time, logging, ast
import pandas as pd
from tqdm import tqdm as tdqm
from torch.utils.data import Dataset, DataLoader
logging.disable(logging.CRITICAL)  # Disables all logging calls of severity 'CRITICAL' and below
from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
from models.llavamodel.llava.llava.eval.run_llava import eval_model

start_time = time.time()

output_dir = "output_results"
# Delete directory if it exists and create a new one
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

data = pd.read_csv("data/llava_data.csv")

# select last 20 rows for testing
data = data[:50]

# data = data[1:10]  # Select only first 10 rows for testing
# sort by country and then income
data = data.sort_values(by=['country', 'income'], ascending=[True, True], ignore_index=True)

class Go_WVS_Img_Dataset(Dataset):
    def __init__(self, data):
        self.img_id = data['image_id']
        self.image_path = data['image_path']
        self.country = data['country']
        self.income = data['income']
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
            'income': self.income[idx],
            'question_text': self.question_text[idx],
            'country_prompt': self.country_prompt[idx],
            'generic_prompt': self.generic_prompt[idx],
            'option_labels': self.option_labels[idx],
            'full_options': self.full_options[idx],
            'selection_answers': self.selection_answers[idx]
        }
        
eval_dataset = Go_WVS_Img_Dataset(data)
dataset_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=1)
model_path = "liuhaotian/llava-v1.5-7b"


def evaluate_model(prompts_batch, img_files_batch, letter_options, full_options):
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

# Loop through the dataset and evaluate each batch
results_dict = {}; combined_results = {}
for i, batch in tdqm(enumerate(dataset_loader), total=len(dataset_loader)):
    prompts_batch, img_files_batch, letter_options, full_options = batch['generic_prompt'], batch['image_path'], batch['option_labels'], batch['full_options']

    # convert selection_answers to list of floats
    batch['selection_answers'] = [ast.literal_eval(each_sublabel.strip()) for each_sublabel in batch['selection_answers']]
    # Pass the batched data to the evaluation function
    batch_results = evaluate_model(prompts_batch, img_files_batch, letter_options, full_options)

    # Store the results of batche values and batch results
    batch_dict = {**batch, **batch_results}

    # Store the results of batch_dict in results_dict
    for key, value in batch_dict.items():
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(value)

# Combine the results of all batches
from tqdm import tqdm
for key, value in tqdm(results_dict.items()):
    combined_results[key] = [item for sublist in value for item in sublist]


# Save the results to a csv file
combined_results_df = pd.DataFrame(combined_results)

# Put 'selection_answers' in the right side of 'prob_percent_values'
selection_answers = combined_results_df['selection_answers']
prob_percent_values = combined_results_df['prob_percent_values']
# move selection_answers to the right side of prob_percent_values
combined_results_df.drop(columns=['selection_answers', 'prob_percent_values'], inplace=True)
combined_results_df['prob_percent_values'] = prob_percent_values
combined_results_df['selection_answers'] = selection_answers


print(f"length of results_df: {len(combined_results_df)}")
output_file = os.path.join(output_dir, "results.csv")
# delete file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
combined_results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")




