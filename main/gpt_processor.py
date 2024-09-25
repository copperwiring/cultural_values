import csv, re, os
import base64
import pandas as pd
from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
from models.llavamodel.llava.llava.eval.run_llava import eval_model
from tqdm import tqdm

api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

class LLAVAProcessor:
    def __init__(self, img_files_paths, wvs_questions, wvs_selection, wvs_options):
        self.img_files_paths = sorted(img_files_paths)
        self.wvs_questions = wvs_questions
        self.wvs_options = wvs_options
        self.wvs_selection = wvs_selection
        self.model_path = "liuhaotian/llava-v1.5-7b"


    def make_prompt(self, n, country_name=None):
        if country_name:
            prompt = f"""How would someone from {country_name}  answer the following question: {self.wvs_questions[n]} \n
                        Here are the options: \n
                        {self.wvs_options[n]} \n
                        ASSISTANT: If had to select one of the options, my answer would be"""
        else:
            prompt = f"""Assume the persona of the demography as seen in the image. Then answer the following question: {self.wvs_questions[n]} \n
                    Here are the options: \n
                    {self.wvs_options[n]} \n
                    ASSISTANT: If had to select one of the options, my answer would be"""
            
        option_letter = re.findall(r'\((.)\)', self.wvs_options[n])
        return prompt, option_letter, self.wvs_options[n]

    # Instread of processing data with number of files, we should do for each country in the selection
    # Reason, we want to pass images based on country name.
    # We have questions and options for each country, so we can pass them as well.
    def process_data(self, image_data, use_images, use_country_name, test=True):
        data = []
        print("test", test)
        questions = self.wvs_questions[15:20] if test else self.wvs_questions
        for n in tqdm(range(len(questions))):

            # # TO CHECK:
            # UPDATE COUNTRIES TO PROCESS BECAUSE WE DONT NEED ALL DOLLARSTREET COUNTRIES FOR ALL QUESTIONS
            # IT DEPENDS ON THE QUESTION
            self.wvs_selection[n] = eval(self.wvs_selection[n].replace("defaultdict(<class 'list'>, ", "").rstrip(")")) \
                                                    if "defaultdict" in self.wvs_selection[n] \
                                                    else self.wvs_selection[n]
        
            # find all countries in the selection from image_data which are in the selection
            dollar_street_countries = image_data['country.name'].unique() 

            # dollar_street_countries = ['Egypt', 'Ethopia'] if test else dollar_street_countries
            countries_to_process = [country for country in dollar_street_countries if country in self.wvs_selection[n]]

            for dollar_street_country in countries_to_process:
                all_file_paths = image_data[image_data['country.name'] == dollar_street_country]['imageFullPath'].values
                all_image_ids = image_data[image_data['country.name'] == dollar_street_country]['id'].values
               
                for each_img_file_path, each_image_id in zip(all_file_paths, all_image_ids):
                    img_file_path = each_img_file_path if use_images else None
                    image_id = each_image_id 

                    # img_file_path = image_data[image_data['country.name'] == dollar_street_country]['imageFullPath'].values[0] if use_images else None
                    # image_id = image_data[image_data['country.name'] == dollar_street_country]['id'].values[0] if use_images else None

                    wvs_country_distribution = self.wvs_selection[n][dollar_street_country]

                    ques_id = n # this is the index of the question

                    prompt, letter_options, full_options = self.make_prompt(n, dollar_street_country) if use_country_name else self.make_prompt(n)


        return data


    def format_result(self, ques_idx, question, image_id, country, use_images,\
                    use_country_name, img_path, \
                    top10_token_prob, result_prob_percent_dict, result_prob_option_dict, \
                    wvs_distribution, prompt):
        result_formatted = {
            "useimage": use_images, # was image used with the prompt
            "usecountry": use_country_name, # was country name used in the prompt
            "ques_idx": ques_idx,
            "question": question,
            "image_id": image_id,
            "img_path": img_path,
            "country": country,
            "top10_token_prob": top10_token_prob,
            "options_prob_percent": result_prob_percent_dict,
            "options_prob": result_prob_option_dict,
            "wvs_distribution": wvs_distribution,
            "prompt": prompt
        }
        return result_formatted

    def save_results(self, data, file_name):
        with open(file_name, mode='w', newline='') as file:
            fieldnames = ["useimage", 
                          "usecountry", 
                          "ques_idx", 
                          "question", 
                          "image_id",
                          "img_path",
                          "country",
                          "top10_token_prob",
                          "options_prob_percent",
                          "options_prob",
                          "wvs_distribution",
                          "prompt"
                            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
