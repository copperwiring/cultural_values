import argparse
import torch
import torch.nn.functional as F

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
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from collections import defaultdict 


def image_parser(args):
    out = args.image_file.split(args.sep)
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

def get_prompt(args, model):
    qs = args.query
    img_file = args.image_file
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if img_file:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    else:
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = qs
    return qs

def get_conv_mode(model_name):
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    elif "mistral" in model_name.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"
    
def get_prob_percent(token_prob_options):
    total_prob = sum(prob for _, prob in token_prob_options)
    # normalized_probabilities = {option: round(prob.item() / total_prob.item(), 2) for option, prob in token_prob_options}
    # calculate prob at % til 2 decimal places
    prob_percent = {option: round(prob.item() / total_prob.item(), 2)*100 for option, prob in token_prob_options}
    
    # print("*" * 50)
    # print(f"Ranked options:{token_prob_options} by their normalized probabilities in %:")
    # for option, prob in normalized_probabilities.items():
    #     print(f"{option}: {prob * 100:.2f}%")

    # print(f"Sum of probabilities in %: {sum(normalized_probabilities.values()) * 100:.2f}%")

    return prob_percent
    
def eval_model(args, letter_options=None, full_option=None):

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = get_prompt(args, model)

    conv_mode = get_conv_mode(model_name)
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print("*" * 50)
    # print(f"Prompt: \n{prompt}")
    # print("*" * 50)

    if args.image_file:
        image_files = image_parser(args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    else:
        images_tensor = None
        image_sizes = None


    target_prompt = prompt 
    input_ids = tokenizer_image_token(target_prompt, 
                                        tokenizer, 
                                        IMAGE_TOKEN_INDEX, 
                                        return_tensors="pt"
                                        ).unsqueeze(0).cuda()
    
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model.forward(
            input_ids=input_ids, 
            images=None if images_tensor is None else images_tensor,
            image_sizes=image_sizes,
            attention_mask=attention_mask
            )
    
    logits = outputs.logits[:, -1, :]  # Get the logits for the last token position
    probabilities = F.softmax(logits, dim=-1).squeeze()

    # Get all predicted tokens and their probabilities
    all_probs, all_indices = torch.topk(probabilities, probabilities.size(0))
    all_tokens = tokenizer.convert_ids_to_tokens(all_indices)
    
    # Get top 10 predicted tokens and their probabilities
    top_10probs, top_5indices = torch.topk(probabilities, 10)
    top_10tokens = tokenizer.convert_ids_to_tokens(top_5indices)

    top10_token_prob = {token: prob for token, prob in zip(top_10tokens, top_10probs)}
    top10_token_prob = [(token, prob.item()) for token, prob in top10_token_prob.items()]


    # print("Top 10 predicted tokens, their index and their probabilities:")
    # for token, index, prob in zip(top_5tokens, top_5indices, top_5probs):
    #     print(f"{token}: {index}:{prob:.20f}")   


    options = letter_options
    token_prob_options = {token: probabilities[tokenizer.convert_tokens_to_ids(token)] for token in options}
    # token_prob_options = sorted(token_prob_options.items(), key=lambda x: x[1], reverse=True)
    token_prob_options = [(token, prob) for token, prob in token_prob_options.items()]
    # token_ids = {tokenizer.convert_tokens_to_ids(token): prob for token, prob in token_prob_options}
    # print(f"Token, index, probability of options: {options} in vocab:")
    # for token, id, prob in zip(token_prob_options, token_ids.keys(), token_ids.values()):
    #     print(f"{token[0]}, {id}, {prob:.20f}")


    prob_percent = get_prob_percent(token_prob_options)

    return options, token_prob_options, prob_percent, top10_token_prob
           


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
