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
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re, os
import torch
import numpy as np

def image_parser(image_file):
    out = image_file.split(',')
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def eval_model(model_path, image_file, query, options):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    qs = query
    img_file = image_file
    # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs: # IMAGE_PLACEHOLDER is 
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         # qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    #         qs = qs
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if img_file:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    else:
        if IMAGE_PLACEHOLDER in qs:  # IMAGE_PLACEHOLDER is ?
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image_file is not None:
        image_files = image_parser(image_file)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
    else:
        images_tensor = None
        image_sizes = None

    log_lik_scores = []

    for option in options:

        target_prompt = prompt + ' ' + option
        print(target_prompt)

        input_ids = (
            tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.forward(
                input_ids=input_ids,
                labels=input_ids,
                attention_mask=attention_mask,
                images=images_tensor,
                )

        log_lik_scores.append((option, -outputs.loss.item()))

    pred_id = np.argmax(np.asarray([x[1] for x in log_lik_scores]))
    print(log_lik_scores)
    # print('Prediction: {}'.format(log_lik_scores[pred_id]))
    # Print the log likelihood scores in new lines
    print("Log likelihood scores:")
    for option, score in log_lik_scores:
        print(f"{option}: {score}")

    # Convert log likelihood to probabilities using softmax
    scores = np.array([score for _, score in log_lik_scores]) # 
    probabilities = softmax(scores)

    print("*"*50)
    print("Probabilities:")
    # Print each option with its corresponding probability
    for option, probability in zip(options, probabilities):
        print(f"{option}: {probability}")

    # Find the option with the highest probability
    pred_id = np.argmax(probabilities)
    print(f"Prediction: {options[pred_id]} with probability {probabilities[pred_id]}")

    # Sum to check if the probabilities sum to 1
    print(f"Sum of probabilities: {np.sum(probabilities)}")

    print("*"*50)
    print(f"Image file: {image_file}")



if __name__ == '__main__':    

    model_path = "liuhaotian/llava-v1.5-7b"

    base_dir = "data/dollarstreet/assets/"
    sub_dir_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
    # choose a random sub directory
    randm_num = np.random.randint(0, len(sub_dir_list))
    sub_dir = sub_dir_list[randm_num]
    sub_dir_path = os.path.join(base_dir, sub_dir)
    # image_file = sub_dir_path + '/' + sub_dir_path.split('/')[-1] + '.jpg'

    image_file = "data/dollarstreet/assets/5d4bde20cf0b3a0f3f3359f7/5d4bde20cf0b3a0f3f3359f7.jpg"

    prompt = """Human: We are going to do object detection. What do you see in the image? \n\n                
    Here are the options: \n\n                    
    [\'(A) People\', 
    \'(B) Cat\', 
    \'(C) Dog\', 
    \'(D) Building\', 
    \'(E) Don\'t know\', 
    \'(F) No answer\'
    \'(G) Invalid option\' \n\n
    """    

    # prompt = """Human: How satisfied are you with the following? \n\n
    # The way the local authorities are solving the region’s affairs \n\n                    
    # Here are the options: \n\n                    
    # [\'(A) Completely dissatisfied\', 
    # \'(B) Rather dissatisfied\', 
    # \'(C) Rather satisfied\', 
    # \'(D) Completely satisfied\', 
    # \'(E) Don\'t know\', 
    # \'(F) No answer\'
    # \'(G) Invalid option\' \n\n
    # """    

    shared_prompt = """Assistant: If had to select one of the options, my answer would be ("""

    options = [shared_prompt+x for x in  ['A', 'B', 'C', 'D', 'E', 'F', 'G']]

    # options = [shared_prompt+x for x in  ['A) Completely dissatisfied', 'B) Rather dissatisfied', 'C) Rather satisfied', 'D) Completely satisfied', "E) Don't know", 'F) No answer']]

    eval_model(model_path, image_file, prompt, options)