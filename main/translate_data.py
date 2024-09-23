from main.create_dataset.data_extractor import DataLoader, DataProcessor
from llava_processor import LLAVAProcessor
import os, shutil
import pandas as pd
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

import logging
logging.disable(logging.CRITICAL)  # Disables all logging calls of severity 'CRITICAL' and below


data_loader = DataLoader("Anthropic/llm_global_opinions", "data/dollarstreet/images_v2.csv")
questions, selections, options = data_loader.get_wvs_data()
dollarstreet_data = data_loader.get_dollarstreet_data()

print(f"Number of question in original WVS data: {len(questions)}")

data_processor = DataProcessor(selections)
common_countries = data_processor.filter_common_countries(dollarstreet_data['country.name'].unique())

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

def make_prompt(n, country_name=None):
    if country_name:
        prompt = f"""Human: How would someone from {country_name}  answer the following question: {questions[n]} \n
                    Here are the options: \n
                    {options[n]} \n
                    Assistant: If had to select one of the options, my answer would be ( """
    else:
        prompt = f"""Human: {questions[n]} \n
                Here are the options: \n
                {options[n]} \n
                Assistant: If had to select one of the options, my answer would be ( """
        
    return prompt

tranlated_questions_with_country = {}
tranlated_questions_without_country = {}
# For each country, we will translate tall questions to the language of the country
def translate_questions(translated_questions, use_country_name):
    for i, country in enumerate(common_countries):
        for question in questions[:2]:
            print(f"Processing data for {country}...")
            prompt_ques = make_prompt(i, country) if use_country_name else make_prompt(i)
            input_ids = tokenizer(f"translate English to {country}: {prompt_ques}", return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            translated_questions[country] = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_questions

translate_questions(tranlated_questions_with_country, use_country_name=True)
translate_questions(tranlated_questions_without_country, use_country_name=False)

# Create a dataframe to save the translated questions: both with and without country name
# 3 columns: country, question, translation (with country name) and translation (without country name)
df = pd.DataFrame(columns=["country", "question", "translation_with_country", "translation_without_country"])
for country in common_countries:
    for i, question in enumerate(questions[:2]):
        df = df.append({"country": country, "question": question, "translation_with_country": tranlated_questions_with_country[country], "translation_without_country": tranlated_questions_without_country[country]}, ignore_index=True)

df.to_csv("output/translated_questions.csv", index=False)

