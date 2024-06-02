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

        option_probabilities = {}
        option_logits = {}

        print(f"Options: {options}")
        print("*" * 50)
        for option in options:
            target_prompt = prompt + ' ' + option
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
            option_logits[option] = logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            option_index = self.tokenizer.convert_tokens_to_ids(option)
            option_probabilities[option] = probabilities[option_index].item()

            # Get the top 10 predicted tokens and their probabilities
            top_probs, top_indices = torch.topk(probabilities, 5)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices)

            print(f"\nOption: '{option}':")
            print(f"Probability of '{option}': {option_probabilities[option]:.12f}")
            print("Top 5 predicted tokens (i.e. last token) and probabilities:")
            for token, prob in zip(top_tokens, top_probs):
                print(f"{token}: {prob.item():.12f}")        

        sorted_options = sorted(option_probabilities.items(), key=lambda x: x[1], reverse=True)
        self.print_ranked_options(sorted_options)

    def print_ranked_options(self, sorted_options):
        print("*" * 50)
        print("Ranked options by their probabilities in vocab:")
        for option, probability in sorted_options:
            print(f"{option}:  {probability:.12f}")

        total_prob = sum(prob for _, prob in sorted_options)
        normalized_probabilities = {opt: prob / total_prob for opt, prob in sorted_options}

        print("*" * 50)
        print("Ranked options by their normalized probabilities:")
        for option, probability in sorted_options:
            print(f"{option}: {normalized_probabilities[option] * 100:.2f}%")

        print("-" * 50)
        print(f"Sum of probabilities in %: {sum(normalized_probabilities.values()) * 100:.2f}%")
        print("*" * 50)

if __name__ == '__main__':
    model_path = "liuhaotian/llava-v1.5-7b"
    evaluator = LLavaModelEvaluator(model_path)

    # -----------------------------------------
    base_dir = "data/dollarstreet/assets/"
    sub_dir_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
    randm_num = np.random.randint(0, len(sub_dir_list))
    sub_dir = sub_dir_list[randm_num]
    sub_dir_path = os.path.join(base_dir, sub_dir)
    
    image_file_ds = "data/dollarstreet/assets/5d4bde20cf0b3a0f3f3359f7/5d4bde20cf0b3a0f3f3359f7.jpg"
    prompt_ds = """Human: How satisfied are you with the following? \n\n
    The way the local authorities are solving the region’s affairs \n\n                    
    Here are the options: \n\n                    
    ['(A) Completely dissatisfied', 
    '(B) Rather dissatisfied', 
    '(C) Rather satisfied', 
    '(D) Completely satisfied', 
    '(E) Don\'t know', 
    '(F) No answer'
    '(G) Invalid option'] \n\n
    Assistant: If had to select one of the options, my answer would be """

    options_ds = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # -----------------------------------------
    image_file = "https://cdn.pixabay.com/photo/2023/08/18/15/02/dog-8198719_640.jpg"
    # image_file = ""
    # image_file = "https://llava-vl.github.io/static/images/view.jpg"
    prompt = """What object is in the image? The image has a \n\n"""
    # shared_prompt = 'This is an image of a: '
    # options = ['balloon', 'potato', 'river', 'hands', 'umbrella']
    options = ['monkey', 'tree', 'water', 'animal', 'dog']
    

    # ds =  dollarstory. 
    # Set to True to use the ds version inputs. 
    # Set to False to use the image classification version inputs for sanity check.
    use_ds_version = False  

    if use_ds_version:
        evaluator.eval_model(image_file_ds, prompt_ds, options_ds)
    else:
        evaluator.eval_model(image_file, prompt, options)
