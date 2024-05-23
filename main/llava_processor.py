import csv
import pandas as pd
from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
from models.llavamodel.llava.llava.eval.run_llava import eval_model
from tqdm import tqdm

class LLAVAProcessor:
    def __init__(self, img_files_paths, wvs_questions, wvs_options):
        self.img_files_paths = sorted(img_files_paths)
        self.wvs_questions = wvs_questions
        self.wvs_options = wvs_options
        self.model_path = "liuhaotian/llava-v1.5-7b"

    def make_prompt(self, n, country_name=None):
        if country_name:
            prompt = f"""Human: How would someone from {country_name}  answer the following question: {self.wvs_questions[n]} \n
                        Here are the options: \n
                        {self.wvs_options[n]} \n
                        Assistant: If had to select one of the options, my answer would be ( """
        else:
            prompt = f"""Human: {self.wvs_questions[n]} \n
                    Here are the options: \n
                    {self.wvs_options[n]} \n
                    Assistant: If had to select one of the options, my answer would be ( """
            
        return prompt

    # Instread of processing data with number of files, we should do for each country in the selection
    # Reason, we want to pass images based on country name.
    # We have questions and options for each country, so we can pass them as well.
    def process_data(self, image_data, num_of_files, use_images, use_country_name):
        data = []
        for i in tqdm(range(num_of_files)):
            # choose country name from ith row of the image data
            dollar_street_country = image_data.iloc[i]['country.name']
            img_file_path = self.img_files_paths[i] if use_images else None
            image_id = image_data[image_data['imageFullPath'] == img_file_path]['id'].values[0] if use_images else None

            for n in range(len(self.wvs_questions)):
                set_index = n
                prompt = self.make_prompt(n, dollar_street_country) if use_country_name else self.make_prompt(n)
                result = self.evaluate_model(prompt, img_file_path)
                data.append(self.format_result(set_index, image_id, dollar_street_country, use_images, use_country_name, img_file_path, prompt, result))
        
        return data

    def evaluate_model(self, prompt, img_file):
        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": img_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
            })()
        self.result = eval_model(args)
        return self.result

    def format_result(self, set_index, id, country_name, use_images, use_country_name, img_file, question, result):
        result_formatted = {
            "set_index": set_index,
            f"used_images?_use_images_{use_images}_use_country_name_{use_country_name}": use_images,
            f"country?_use_images_{use_images}_use_country_name_{use_country_name}": country_name if country_name else None,
            f"id?_use_images_{use_images}_use_country_name_{use_country_name}": id,
            f"img_path?_use_images_{use_images}_use_country_name_{use_country_name}": img_file,
            f"question?_use_images_{use_images}_use_country_name_{use_country_name}": question,
            f"llava_output?_use_images_{use_images}_use_country_name_{use_country_name}": result
        }
        return result_formatted

    def save_results(self, data, file_name, use_images, use_country_name):
        with open(file_name, mode='w', newline='') as file:
            fieldnames = ['set_index', \
                           f'used_images?_use_images_{use_images}_use_country_name_{use_country_name}', \
                           f'country?_use_images_{use_images}_use_country_name_{use_country_name}', \
                           f'id?_use_images_{use_images}_use_country_name_{use_country_name}', \
                           f'img_path?_use_images_{use_images}_use_country_name_{use_country_name}', \
                           f'question?_use_images_{use_images}_use_country_name_{use_country_name}', \
                           f'llava_output?_use_images_{use_images}_use_country_name_{use_country_name}'
                           ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
