## GPT4-o code to get agree, disaqgree for a question based on image and it log probability
from openai import OpenAI
import pandas as pd
from PIL import Image
import requests
import io, os , ast
from io import BytesIO
import base64, shutil
import numpy as np
from get_prompt import rules, key_names, make_prompt, suffix_txt

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DataPreparation():
    """This class is responsible for preparing the data for the model"""
   
    def __init__(self, question, options, option_list):
        self.question = question
        self.options = options
        self.option_list = option_list

    def image_to_base64(self, image_path):
       """
       This function converts an image to a base64 string
       """
       with open(image_path, "rb") as f:
           image = base64.b64encode(f.read()).decode('utf-8')
           return image
       
    def get_prompt(self):
        """
        This function gets a question and options
        """
        
        # Create the dictionary with keys based on the length of the option_list, values are always "<option>"
        choice_dict = {key_names[i]:"<option>" for i in range(min(len(self.option_list), len(key_names)))}
        prompt = make_prompt(self.question, self.options, choice_dict)

        return prompt

# image_url = "https://anthrowcircus.com/wp-content/uploads/2022/10/young-women-in-bunads.jpg"
# img = requests.get(image_url)

class gpt4o():
    def __init__(self):
        self.rules = rules

    def gpt_response(self, prompt, base64_image):
        # Create the API request
        error = False
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the correct model for your needs
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.rules}]
                    },
                    {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url":
                                    {"url": f"data:image/png;base64,{base64_image}"
                                    }
                                    }
                                ]
                    },

                ],
                # max_tokens=300,
                logprobs=True,  # Set logprobs to True to get log probabilities
                top_logprobs=20
            )
        except Exception as e:
            response = e
            error = True

        return response, error
    
    def get_logprobs(self, prompt, base64_image, normalized_option_list):

        response, error = self.gpt_response(prompt, base64_image)
    
        if error:
            output_message = "Error occurred while fetching the response"
            top_k_values = [(option, 0) for option in normalized_option_list]
            return output_message, top_k_values
        else:
            # Extract the response from the model
            choice = response.choices[0]
            output_message = choice.message.content  # Extract the content of the response

            # Print log probabilities for each token in the response
            top_k_values = []
            for token_logprob in choice.logprobs.content:
                token = token_logprob.token
                if token in option_list:
                    top_k = token_logprob.top_logprobs
                    # print token and logprob values from list 
                    top_k_values = [(details.token, details.logprob) for details in top_k if details.token.strip() in normalized_option_list and details.token == details.token.strip()]
                    print(f"topk k values: {top_k_values}")
                    break
            if not top_k_values:
                top_k_values = [(option, 0) for option in normalized_option_list]
            else:
                top_k_values = top_k_values

            return output_message, top_k_values
    
if __name__ == "__main__":

    data_dir = "/home/vsl333/cultural_values/created_csv_data"
    out_dir = "/home/vsl333/cultural_values/gpt_csv_data"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csvfile_list = os.listdir(data_dir)

    # exclude file that ends with "ds_people.csv" file
    csvfile_list = [file for file in csvfile_list if not file.endswith("ds_people.csv")]

    one_csv_file = csvfile_list[0]
    df = pd.read_csv(os.path.join(data_dir, one_csv_file))
    df = df[df['category']=='Traditions / art / history'] # Filter the data based on category
    # breakpoint()
    print(f"Processing {one_csv_file} file")
    print(f"Total rows: {len(df)}")
    # Initialize the columns with None or any default value
    df['top_k'] = None
    df['output_txt'] = None
    df['top_k_prob'] = None

    for row_id, row in df.iterrows():
        question = row['question_text']
        options = row['full_options']
        option_list = ast.literal_eval(row['option_labels'])

        image_path = row['image_path']

        data_preparation = DataPreparation(question, options, option_list)
        base64_image = data_preparation.image_to_base64(image_path)
        prompt = data_preparation.get_prompt()
        print("image_path: ", image_path)
        print("option_list: ", option_list)
        # breakpoint()

        gpt4o_obj = gpt4o()
        normalized_option_list = [option.upper() for option in option_list]
        output_txt, top_k = gpt4o_obj.get_logprobs(prompt, base64_image, normalized_option_list)

        # Initialize the counter and max_attempts
        counter = 0
        max_attempts = 6
        # Loop until you find a non-zero value or reach max_attempts
        while all(value == 0 for _, value in top_k) and counter < max_attempts:
            
            if counter == 0:
                suffix_txt_choice = "You are allowed to make assumptions from your knowledge"
            else:
                suffix_txt_choice = "Don't say " + str(output_txt) + np.random.choice(suffix_txt)
            output_txt, top_k = gpt4o_obj.get_logprobs(prompt + suffix_txt_choice, base64_image, option_list)
            
            print(f"Attempt {counter + 1}: suffix_txt used - {suffix_txt_choice}")
            
            # Increment the counter
            counter += 1

        # breakpoint()
        # Check why the loop exited
        if all(value == 0 for _, value in top_k):
            # print("Max attempts reached with all values still being zero.")
            top_k = [(option, 0) for option in normalized_option_list]
        else:
            # print("Non-zero value found in top_k!")
            top_k = top_k
        
        # Sort the top_k values based on key 

        top_k = sorted(top_k, key=lambda x: x[0])
        # convert topk to probability
        sum_topk = sum([value for _, value in top_k])
        # Normalize values or set them to 0 if sum_topk is 0
        if sum_topk != 0:
            top_k_prob = [(option, value / sum_topk) for option, value in top_k]
        else:
            top_k_prob = [(option, 0) for option, _ in top_k]  # Assign 0 if sum_topk is 0
        top_k_prob_percent = [(option, round(value*100, 2)) for option, value in top_k_prob]
        # Sort based on alphabetical order
        top_k_pp_values = [value for _, value in top_k_prob_percent]

        # print(f"prompt: {prompt}")
        # print(f"output_txt: {output_txt}")
        # print(f"top_k: {top_k}")
        print(f"top_k_prob: {top_k_prob_percent}")

        # create a new column in the dataframe and save the output_txt and top_k values
        df.at[row_id, 'output_txt'] = output_txt
        df.at[row_id, 'top_k'] = top_k
        df.at[row_id, 'top_k_p'] = top_k_prob_percent
        df.at[row_id, 'top_k_pv'] = top_k_pp_values

        # write to csv file
        with open(os.path.join(out_dir, f"{one_csv_file}_gpt_output.csv"), "a") as f:
            df.to_csv(f, index=False, header=False) if row_id > 0 else df.to_csv(f, index=False, header=True)

    # # Save the dataframe with the new columns
    # df.to_csv(os.path.join(out_dir, f"{one_csv_file}_gpt_output.csv"), index=False)






