"""
This script is used to run LLAVA(Vilcuna) on a set of images and save the results to a CSV file.
"""
import os
import json
import csv
import pandas as pd

cur_dir = os.getcwd()
root_dir = os.path.dirname(os.path.abspath(cur_dir))
print("HF root directory:", root_dir)

from models.llavamodel.llava.llava.model.builder import load_pretrained_model
from models.llavamodel.llava.llava.mm_utils import get_model_name_from_path
from models.llavamodel.llava.llava.eval.run_llava import eval_model

class LLAVAProcessor:
    """
    A class to process data using the LLAVA model.

    Attributes:
        base_dir (str): The base directory where LLAVA model and data are located.
    """
    
    def __init__(self, img_files_paths, wvs_questions, wvs_options):
        """
        Initializes the LLAVAProcessor with the specified base directory and changes
        the working directory to the base directory.

        Parameters:
            prompt (str): The prompt to use for the model.
            img_files_paths (list): The list of image file paths
            wvs_questions (list): The list of WVS questions
            wvs_options (list): The list of WVS options
        """

        self.img_files_paths = img_files_paths
        self.wvs_questions = wvs_questions[:2]
        self.wvs_options = wvs_options[:2]
        self.model_path = "liuhaotian/llava-v1.5-7b"


    def load_data(self):
        """
        Loads image files and text data from predefined directories into memory.
        """

        # Sort image files to ensure consistency
        self.img_files_path = sorted(self.img_files_path)


    def make_prompt(self, n):

        prompt = f"""Human: {self.wvs_questions[n]} \n

                  Here are the options: \n
                  {self.wvs_options[n]} \n

                  Assistant: If had to select one of the options, my answer would be ( """

        return prompt

    def process_data(self, image_data, num_of_files, use_images):
        """
        Processes the data based on the number of samples, whether to use images, and text choice.

        Parameters:
            id (str): The UUID of the data sample.
            num_of_files (int): The number of files to process.
            use_images (bool): If True, images will be used in the evaluation.
        
        """
        data = []

        for i in range(num_of_files):
            print("I am processing file number: ", i)
            img_file_path = self.img_files_paths[i] if use_images else None
            image_id = image_data[image_data['imageFullPath'] == img_file_path]['id']
            image_id = image_id.values[0]
        
            len_questions = len(self.wvs_questions)
            len_options = len(self.wvs_options)
            assert len_questions == len_options, "All data samples must have the same number of questions and options."

            for n in range(len_questions):
                question = self.wvs_questions[n]
                prompt = self.make_prompt(n)
                result = self.evaluate_model(prompt, img_file_path)
                data.append(self.format_result(image_id, use_images, img_file_path, question, result))
        
        return data

    def evaluate_model(self, prompt, img_file):
        """
        Evaluates the model with the provided prompt and image file.

        Parameters:
            prompt (str): The prompt to use for the model.
            img_file (str|None): The path to the image file, if any.

        Returns:
            dict: The result of the model evaluation.
        """

        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": img_file,
            "sep": ",",
            "temperature": 0, # Set to 0 to disable randomness
            "top_p": None, # TO FIX: I get user warning: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `None` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
            "num_beams": 1,
            "max_new_tokens": 512
            })()
        
        print(f"image_file: {img_file}")
        self.result = eval_model(args)
        
        return self.result

    def format_result(self, id, use_images, img_file, question, result):
        # image_id, use_images, img_file_path, result
        """
        Formats the evaluation results into a dictionary suitable for JSON"
        """
        
        # TO DO: make some parameters self attributes
        result_formatted = {
            "Used Images?": use_images,
            "id": id,
            "img_path": img_file,
            "question": question,
            "llava_output": result
        }
    
        return result_formatted

    def save_results(self, data, file_name):

        with open(file_name, mode='w', newline='') as file:
            fieldnames = ['Used Images?', 'id', 'img_path', 'question', 'llava_output']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)










