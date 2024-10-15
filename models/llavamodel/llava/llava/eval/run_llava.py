import argparse
import ast
import re
from io import BytesIO

import requests
import torch
import torch.nn.functional as F
from PIL import Image

from models.llavamodel.llava.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.llavamodel.llava.llava.conversation import conv_templates
from models.llavamodel.llava.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from models.llavamodel.llava.llava.model.builder import load_pretrained_model
from models.llavamodel.llava.llava.utils import disable_torch_init


def image_parser(image_file_str, sep):
    return image_file_str.split(sep)


def load_image(image_file):
    if image_file.startswith(("http", "https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    return [load_image(image_file) for image_file in image_files]


def get_prompt(args, model):
    qs = args.query
    img_file = args.image_file
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if img_file:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    else:
        if IMAGE_PLACEHOLDER in qs:
            replacement_token = (
                image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            )
            qs = qs.replace(IMAGE_PLACEHOLDER, replacement_token)
        elif model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
    return qs


def get_conv_mode(model_name):
    model_name_lower = model_name.lower()
    if "llama-2" in model_name_lower:
        return "llava_llama_2"
    elif "mistral" in model_name_lower:
        return "mistral_instruct"
    elif "v1.6-34b" in model_name_lower:
        return "chatml_direct"
    elif "v1" in model_name_lower:
        return "llava_v1"
    elif "mpt" in model_name_lower:
        return "mpt"
    else:
        return "llava_v0"


def get_prob_percent(token_prob_alloptions, len_letter_option):
    token_prob_options = token_prob_alloptions[:len_letter_option]
    total_prob = sum(prob for _, prob in token_prob_options)
    if total_prob == 0:
        return {option: 0.0 for option, _ in token_prob_options}
    prob_percent = {
        option: round((prob / total_prob) * 100, 2) for option, prob in token_prob_options
    }
    return prob_percent


def left_pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    if len(sequence) >= max_length:
        return sequence
    padding = torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype)
    return torch.cat([padding, sequence])


def eval_model(
    args,
    prompts_batch,
    img_files_batch=None,
    letter_options=None,
    full_options=None,
    tokenizer=None,
    model=None,
    image_processor=None,
    model_name=None,
):
    conv_mode = get_conv_mode(model_name)
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto-inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    batched_prompts = []
    for prompt in prompts_batch:
        args.query = prompt
        qs = get_prompt(args, model)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        batched_prompts.append(conv.get_prompt())

    tokenized_prompts = [
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0)
        for prompt in batched_prompts
    ]

    max_len = max(len(tokenized_prompt) for tokenized_prompt in tokenized_prompts)
    padded_tokenized_ids = [
        left_pad_sequence_to_max_length(
            tokenized_prompt, max_len, padding_value=tokenizer.pad_token_id
        )
        for tokenized_prompt in tokenized_prompts
    ]
    batched_input_ids = torch.stack(padded_tokenized_ids).to(model.device)

    if img_files_batch:
        images = []
        for img_files_str in img_files_batch:
            image_files = image_parser(img_files_str, args.sep)
            print(f"image_files: {image_files}")
            images.extend(load_images(image_files))
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        image_sizes = [img.size for img in images]
    else:
        images_tensor = None
        image_sizes = None

    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model(
            input_ids=batched_input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
        )

    logits = outputs.logits[:, -1, :]
    probabilities = F.softmax(logits, dim=-1)

    batch_results = {
        "prompt": [],
        "options": [],
        "top10_token_prob": [],
        "prob_percent_sorted": [],
        "sum_prob_percent_sorted": [],
        "prob_percent_keys": [],
        "prob_percent_values": [],
    }

    all_uppercase = list(map(chr, range(65, 91)))  # ['A', 'B', ..., 'Z']

    for i, (prompt, letter_option) in enumerate(zip(batched_prompts, letter_options)):
        try:
            parsed_options = ast.literal_eval(letter_option)
            len_letter_option = len(parsed_options)
        except (ValueError, SyntaxError):
            print(f"Error parsing letter_option at index {i}: {letter_option}")
            len_letter_option = 0

        probs_for_instance = probabilities[i]  # Tensor of size [vocab_size]
        option_labels = all_uppercase[:len_letter_option]  # ['A', 'B', 'C', ...]

        # For logging purposes, get the top 10 tokens (optional)
        top_k = min(10, probs_for_instance.size(0))
        if top_k > 0:
            top_probs, top_indices = torch.topk(probs_for_instance, top_k)
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
            top_token_probs = [(token, prob.item()) for token, prob in zip(top_tokens, top_probs)]
        else:
            top_token_probs = []

        # Prepare a mapping from labels to token IDs
        label_to_token_ids = {}
        for label in option_labels:
            # Encode the label to get token IDs without adding special tokens
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) == 0:
                print(f"No token IDs found for label '{label}'.")
                label_to_token_ids[label] = []
            else:
                label_to_token_ids[label] = token_ids

        # Compute probabilities for each label directly from the full distribution
        prob_percent = {}
        probs_array = probs_for_instance.detach().cpu().numpy()
        for label in option_labels:
            token_ids = label_to_token_ids.get(label, [])
            label_prob = 0.0
            for token_id in token_ids:
                label_prob += probs_array[token_id]
            prob_percent[label] = label_prob

        # Ensure all labels are present in prob_percent
        for label in option_labels:
            if label not in prob_percent:
                prob_percent[label] = 0.0

        # Normalize probabilities to sum to 100%
        sum_probs = sum(prob_percent.values())
        if sum_probs > 0:
            prob_percent_normalized = {k: (v / sum_probs) * 100 for k, v in prob_percent.items()}
        else:
            prob_percent_normalized = {k: 0.0 for k in prob_percent}

        prob_percent_sorted = {k: prob_percent_normalized[k] for k in sorted(prob_percent_normalized)}
        sum_prob_percent_sorted = sum(prob_percent_sorted.values())
        prob_percent_keys = list(prob_percent_sorted.keys())
        prob_percent_values = list(prob_percent_sorted.values())

        batch_results["prompt"].append(prompt)
        batch_results["options"].append(letter_option)
        batch_results["top10_token_prob"].append(top_token_probs)
        batch_results["prob_percent_sorted"].append(prob_percent_sorted)
        batch_results["sum_prob_percent_sorted"].append(sum_prob_percent_sorted)
        batch_results["prob_percent_keys"].append(prob_percent_keys)
        batch_results["prob_percent_values"].append(prob_percent_values)

    return batch_results
