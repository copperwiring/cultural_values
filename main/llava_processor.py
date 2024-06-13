import csv, re, ast
import pandas as pd
from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
from models.llavamodel.llava.llava.eval.run_llava import eval_model
from tqdm import tqdm

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
            prompt = f"""{self.wvs_questions[n]} \n
                    Here are the options: \n
                    {self.wvs_options[n]} \n
                    ASSISTANT: If had to select one of the options, my answer would be"""
            
        option_letter = re.findall(r'\((.)\)', self.wvs_options[n])
        return prompt, option_letter, self.wvs_options[n]

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
                # TODO:
                # UPDATE COUNTRIES TO PROCESS BECAUSE WE DONT NEED ALL DOLLARSTREET COUNTRIES FOR ALL QUESTIONS
                # IT DEPENDS ON THE QUESTION
                self.wvs_selection[n] = eval(self.wvs_selection[n].replace("defaultdict(<class 'list'>, ", "").rstrip(")")) \
                                                         if "defaultdict" in self.wvs_selection[n] \
                                                        else self.wvs_selection[n]
                
                countries_to_process = dollar_street_country in self.wvs_selection[n]  
                if not countries_to_process:
                    continue
                wvs_country_distribution = self.wvs_selection[n][dollar_street_country]

                ques_id = n # this is the index of the question

                prompt, letter_options, full_options = self.make_prompt(n, dollar_street_country) if use_country_name else self.make_prompt(n)
                options, token_prob_options, prob_percent, top10_token_prob = self.evaluate_model(prompt, \
                                                                                                           img_file_path, \
                                                                                                           letter_options, \
                                                                                                           full_options)
                # self.print_ranked_options(sorted_token_prob_options, options)
                # self.print_ranked_options(sorted_underscore_token_prob_options, options_with_underscore)
                result = [options, token_prob_options, prob_percent, top10_token_prob]

                # norm_prob_options is a dictionary with options as keys and probabilities as values in a tensor e.g {'A': tensor(0.3, device='cuda:0')}
                result_prob_option_dict = {token: prob.item() for token, prob in token_prob_options}
                result_prob_percent_dict = prob_percent
                result_formatted = self.format_result(ques_id, self.wvs_questions[n], 
                                                      image_id, dollar_street_country, 
                                                      use_images, use_country_name, img_file_path, 
                                                      top10_token_prob,
                                                      result_prob_percent_dict,
                                                      result_prob_option_dict,
                                                      wvs_country_distribution,
                                                      prompt)
                data.append(result_formatted)
        return data

    def evaluate_model(self, prompt, img_file, letter_options, full_option):
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
        self.result = eval_model(args, letter_options, full_option)
        return self.result

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
