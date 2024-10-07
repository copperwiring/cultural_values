import argparse
import torch, ast
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


def image_parser(image_file_str, sep):
    out = image_file_str.split(sep)
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
    
def get_prob_percent(token_prob_alloptions, len_letter_option):
    token_prob_options = token_prob_alloptions[:len_letter_option]
    total_prob = sum(prob for _, prob in token_prob_options)
    prob_percent = {option: round((prob / total_prob)*100, 2) for option, prob in token_prob_options}
    
    return prob_percent

# left padding
def left_pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

# right padding
def right_pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length with right padding."""
    if len(sequence) >= max_length:
        return sequence
    # Create padding of appropriate length and append it to the sequence (right padding)
    return torch.cat([sequence, torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype)])
    
def eval_model(args, prompts_batch, img_files_batch=None, letter_options=None, full_options=None, tokenizer=None, model=None, image_processor=None, model_name=None):

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, offload_folder="offload")

    # qs = get_prompt(args, model)

    conv_mode = get_conv_mode(model_name)
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # conv = conv_templates[args.conv_mode].copy()

    batched_prompts = []
    
    # Process prompts in batch
    for prompt in prompts_batch:
        # Set args.query to the specific prompt in the batch
        args.query = prompt

        # Generate the prompt for each input in the batch, with the correct image handling
        qs = get_prompt(args, model)

        # Create a new conversation template for each prompt in the batch
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        # Add the complete prompt for this instance to the batch
        batched_prompts.append(conv.get_prompt())


    # # max length for padding
    max_len = max([len(tokenizer.encode(prompt)) for prompt in batched_prompts])

    tokenizer.padding_side = "left"
    tokenizer.model_max_length = max_len
    
    # Tokenize the batch of prompts
    tokenized_prompts = [
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        for prompt in batched_prompts
    ]

    # Determine the maximum length of input_ids in the batch
    max_len = max([len(tokenized_prompt.squeeze()) for tokenized_prompt in tokenized_prompts])

    # Pad the input_ids to the maximum length
    device = model.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    padded_tokenized_ids= [left_pad_sequence_to_max_length(tokenized_prompt.squeeze(), max_len) for tokenized_prompt in tokenized_prompts]
    batched_input_ids = torch.stack(padded_tokenized_ids).to(model.device)


    # input_ids = torch.cat(tokenized_prompts, dim=0).cuda()

    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()


    # if args.image_file:
    #     image_files = image_parser(args)
    #     images = load_images(image_files)
    #     image_sizes = [x.size for x in images]
    #     images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    # else:
    #     images_tensor = None
    #     image_sizes = None

    # Process images if provided (batch image loading and processing)

    # set device to cpu
    # model.to("cpu")
    if img_files_batch:
        # For each batch, parse image files, load them, and process
        image_files_batch = [image_parser(img_files, args.sep) for img_files in img_files_batch]
        images = [load_images(image_files) for image_files in image_files_batch]
        flat_images = [item for sublist in images for item in sublist]
        images_tensor = process_images(flat_images, image_processor, model.config).to(model.device, dtype=torch.float16)
        image_sizes = [img.size for img in flat_images]
    else:
        images_tensor = None
        image_sizes = None

    # target_prompt = prompt 
    # input_ids = tokenizer_image_token(target_prompt, 
    #                                     tokenizer, 
    #                                     IMAGE_TOKEN_INDEX, 
    #                                     return_tensors="pt"
    #                                     ).unsqueeze(0).cuda()
    
    # attention_mask = torch.ones_like(batched_input_ids)

    # Generate attention mask
    # def generate_attention_mask(padded_sequence, padding_value=tokenizer.pad_token_id):
    #     """Generate attention mask for a padded sequence where 1 is for actual tokens and 0 is for padding."""
    #     return (padded_sequence != padding_value).long()  # Mask: 1 for non-padding, 0 for padding

    # # Generate attention masks for each prompt based on the padding
    # attention_masks = [
    #     generate_attention_mask(padded_prompt) for padded_prompt in padded_tokenized_ids
    # ]

    # Stack the attention masks for the batch
    # batched_attention_mask = torch.stack(attention_masks).to(model.device)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model.forward(
            input_ids=batched_input_ids, 
            images=None if images_tensor is None else images_tensor,
            image_sizes=image_sizes            )
    
    logits = outputs.logits[:, -1, :]  # Get the logits for the last token position
    probabilities = F.softmax(logits, dim=-1).squeeze()

    # Initialize batched prompts empty dict
    batch_results = {
        'prompt': [],
        'options': [],
        'top10_token_prob': [],
        'prob_percent_sorted': [],
        'sum_prob_percent_sorted': [],
        'prob_percent_keys': [],
        'prob_percent_values': []
    }

    # Process each instance in the batch
    for i, (prompt, letter_option) in enumerate(zip(batched_prompts, letter_options)):
        len_letter_option = len(ast.literal_eval(letter_option))
        probs_for_instance = probabilities[i]  # Get probabilities for the i-th instance

        # Get all predicted tokens and their probabilities
        # all_probs, all_indices = torch.topk(probabilities, probabilities.size(0))
        # all_tokens = tokenizer.convert_ids_to_tokens(all_indices)
        
        # Get top 10 predicted tokens and their probabilities
        top_10probs, top_10indices = torch.topk(probs_for_instance, 10)
        top_10tokens = tokenizer.convert_ids_to_tokens(top_10indices)

        top10_token_prob = {token: prob for token, prob in zip(top_10tokens, top_10probs)}
        top10_token_prob = [(token, prob.item()) for token, prob in top10_token_prob.items()]

        # Process the options for evaluation
        # token_prob_options = {token: probs_for_instance[tokenizer.convert_tokens_to_ids(token)] for token in letter_options}
        # token_prob_options = [(token, prob) for token, prob in token_prob_options.items()]
        # # sort by probability
        # token_prob_options = sorted(token_prob_options, key=lambda x: x[1], reverse=True)

        prob_percent = get_prob_percent(top10_token_prob, len_letter_option)

        # check of keys are same as alphabets in cupper case from A - len_letter_option. if not, add the missing keys with 0 value
        for i in range(65, 65+len_letter_option):
            if chr(i) not in prob_percent.keys():
                prob_percent[chr(i)] = 0

        # sort prob_percent dict alphabetically by key
        prob_percent_sorted = {k: prob_percent[k] for k in sorted(prob_percent)}

        # Append the result for this instance in the batch
        batch_results['prompt'].append(prompt)
        batch_results['options'].append(letter_option)
        batch_results['top10_token_prob'].append(top10_token_prob)
        batch_results['prob_percent_sorted'].append(prob_percent_sorted)
        batch_results['sum_prob_percent_sorted'].append(sum(prob_percent_sorted.values()))
        batch_results['prob_percent_keys'].append(list(prob_percent_sorted.keys()))
        batch_results['prob_percent_values'].append(list(prob_percent_sorted.values()))

    # print(f"Batch results: {batch_results}")

    return batch_results
           


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
#     parser.add_argument("--model-base", type=str, default=None)
#     parser.add_argument("--image-file", type=str, required=True)
#     parser.add_argument("--query", type=str, required=True)
#     parser.add_argument("--conv-mode", type=str, default=None)
#     parser.add_argument("--sep", type=str, default=",")
#     parser.add_argument("--temperature", type=float, default=0.2)
#     parser.add_argument("--top_p", type=float, default=None)
#     parser.add_argument("--num_beams", type=int, default=1)
#     parser.add_argument("--max_new_tokens", type=int, default=512)
#     args = parser.parse_args()

#     eval_model(args)
