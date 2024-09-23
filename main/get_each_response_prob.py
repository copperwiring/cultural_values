import os
import re
import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from models.llavamodel.llava.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.llavamodel.llava.llava.conversation import conv_templates, SeparatorStyle
from models.llavamodel.llava.llava.model.builder import load_pretrained_model
from models.llavamodel.llava.llava.utils import disable_torch_init
from models.llavamodel.llava.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

class LLavaModelEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, self.model_name)
        disable_torch_init()

    def image_parser(self, image_file):
        return image_file.split(',')

    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(self, image_files):
        return [self.load_image(image_file) for image_file in image_files]

    def get_prompt(self, img_file, query):
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if img_file:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        else:
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = qs
        return qs

    def get_conv_mode(self):
        if "llama-2" in self.model_name.lower():
            return "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            return "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            return "chatml_direct"
        elif "v1" in self.model_name.lower():
            return "llava_v1"
        elif "mpt" in self.model_name.lower():
            return "mpt"
        else:
            return "llava_v0"

    def eval_model(self, image_file, query, options):
        qs = self.get_prompt(image_file, query)
        conv_mode = self.get_conv_mode()

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print("*" * 50)
        print(f"Prompt: \n{prompt}")
        print("*" * 50)

        if image_file:
            image_files = self.image_parser(image_file)
            images = self.load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        else:
            images_tensor = None
            image_sizes = None


        target_prompt = prompt 
        input_ids = tokenizer_image_token(target_prompt, 
                                            self.tokenizer, 
                                            IMAGE_TOKEN_INDEX, 
                                            return_tensors="pt"
                                            ).unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = self.model.forward(
                input_ids=input_ids, 
                images=None if images_tensor is None else images_tensor,
                image_sizes=image_sizes,
                attention_mask=attention_mask
                )
        
        logits = outputs.logits[:, -1, :]  # Get the logits for the last token position
        probabilities = F.softmax(logits, dim=-1).squeeze()

        # Get all predicted tokens and their probabilities
        all_probs, all_indices = torch.topk(probabilities, probabilities.size(0))
        all_tokens = self.tokenizer.convert_ids_to_tokens(all_indices)
        
        # Get top 10 predicted tokens and their probabilities
        top_5probs, top_5indices = torch.topk(probabilities, 10)
        top_5tokens = self.tokenizer.convert_ids_to_tokens(top_5indices)

        print("Top 10 predicted tokens, their index and their probabilities:")
        for token, index, prob in zip(top_5tokens, top_5indices, top_5probs):
            print(f"{token}: {index}:{prob:.20f}")   


        new_options = ["▁A", "▁B", "▁C", "▁D", "▁E", "▁F", "▁G"]
        new_token_prob_options = {token: probabilities[self.tokenizer.convert_tokens_to_ids(token)] for token in new_options}
        sorted_new_token_prob_options = sorted(new_token_prob_options.items(), key=lambda x: x[1], reverse=True)
        sorted_new_token_ids = {self.tokenizer.convert_tokens_to_ids(token): prob for token, prob in sorted_new_token_prob_options}
        print("*" * 50)
        print(f"Token, index, probability of options: {new_options} in vocab:")
        for token, id, prob in zip(sorted_new_token_prob_options, sorted_new_token_ids.keys(), sorted_new_token_ids.values()):
            print(f"{token[0]}, {id}, {prob:.20f}")

        print("*" * 50)
        token_prob_options = {token: probabilities[self.tokenizer.convert_tokens_to_ids(token)] for token in options}
        sorted_token_prob_options = sorted(token_prob_options.items(), key=lambda x: x[1], reverse=True)
        sorted_token_ids = {self.tokenizer.convert_tokens_to_ids(token): prob for token, prob in sorted_token_prob_options}
        print(f"Token, index, probability of options: {options} in vocab:")
        # sorted_token_prob_options is list of tuples and sorted_token_ids is dict
        for token, id, prob in zip(sorted_token_prob_options, sorted_token_ids.keys(), sorted_token_ids.values()):
            print(f"{token[0]}, {id}, {prob:.20f}")

    
        self.print_ranked_options(sorted_token_prob_options, options)
        self.print_ranked_options(sorted_new_token_prob_options, new_options)
        print("Done")

    def print_ranked_options(self, sorted_options, options):
        # print("*" * 50)
        # print("Ranked options by their probabilities in vocab:")
        # for option, probability in sorted_options:
        #     print(f"{option}:  {probability:.12f}")

        total_prob = sum(prob for _, prob in sorted_options)
        normalized_probabilities = {opt: prob / total_prob for opt, prob in sorted_options}

        print("*" * 50)
        print(f"Ranked options:{options} by their normalized probabilities in %:")
        for option, probability in sorted_options:
            print(f"{option}: {normalized_probabilities[option] * 100:.2f}%")

        print(f"Sum of probabilities in %: {sum(normalized_probabilities.values()) * 100:.2f}%")

if __name__ == '__main__':
    model_path = "liuhaotian/llava-v1.5-7b"
    evaluator = LLavaModelEvaluator(model_path)

    # -----------------------------------------
    base_dir = "data/dollarstreet/assets/"
    sub_dir_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
    randm_num = np.random.randint(0, len(sub_dir_list))
    sub_dir = sub_dir_list[randm_num]
    sub_dir_path = os.path.join(base_dir, sub_dir)
    
    image_file_ds = "https://www.foreigner.bg/wp-content/uploads/2021/01/family-1827369_640.jpg"
    prompt_ds = """This is a photo of people of which country? \n\n                    
    Here are the options: \n\n                    
    [(A) Serbia, 
    (B) Sweden, 
    (C) Romania, 
    (D) Bulgaria, 
    (E) Istanbul
    ] \n\n
    ASSISTANT: If had to select one of the options, my answer would be"""

    # exclude 'F' and 'G' options which are No answer and Invalid option

    # options_ds = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    options_ds = ['A', 'B', 'C', 'D', 'E']

    # options_ds = ['Completely dissatisfied', 'Rather dissatisfied', 'Rather satisfied', 'Completely satisfied', 'Don\'t know', 'No answer', 'Invalid option']
    options_ds = sorted(options_ds)

    # -----------------------------------------
    # image_file = "https://cdn.pixabay.com/photo/2023/08/18/15/02/dog-8198719_640.jpg"
    image_file = "https://i0.wp.com/big-family-small-world.com/wp-content/uploads/2018/10/family-shot-sofia-roman-walls-1-1024x991.jpg?ssl=1"
    # image_file = "https://st4.depositphotos.com/1005381/30046/i/450/depositphotos_300467142-stock-photo-girls-dressed-folk-costumes-church.jpg"
    # image_file = "https://llava-vl.github.io/static/images/view.jpg"
    prompt = """This is a family photo of :\n\n"""
    options = ["dog", "animal", "computer", "river", "▁computer", "▁Computer"]
    options = sorted(options)
    

    # ds =  dollarstory. 
    # Set to True to use the ds version inputs. 
    # Set to False to use the image classification version inputs for sanity check.
    use_ds_version = True  

    if use_ds_version:
        evaluator.eval_model(image_file_ds, prompt_ds, options_ds)
    else:
        evaluator.eval_model(image_file, prompt, options)
