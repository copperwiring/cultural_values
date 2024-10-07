import pandas as pd
import ast, os
from datasets import load_dataset
from collections import defaultdict

class LoadGoDollarstreetCVQAData:
    def __init__(self, go_dataset_path, image_dir, dollarstreet_csv_path = None, cvqa_csv_path = None):
        self.go_dataset = load_dataset(go_dataset_path)
        self.image_dir = image_dir
        self.dollarstreet_data = pd.read_csv(dollarstreet_csv_path) if dollarstreet_csv_path is not None else None
        self.cvqa_data = pd.read_csv(cvqa_csv_path) if cvqa_csv_path is not None else None


    def get_wvs_data(self):
        train_data = self.go_dataset['train']
        train_data_wvs = train_data.filter(lambda x: x['source'] == 'WVS')
        questions = train_data_wvs['question']
        selections = train_data_wvs['selections']
        options = train_data_wvs['options']
        return questions, selections, options

    def get_dollarstreet_data(self):
        return self.dollarstreet_data
    
    def get_cvqa_data(self):
        return self.cvqa_data
    
    


class DataExtractor:
    def __init__(self, wvs_selections):
        self.wvs_countries = self.extract_countries(wvs_selections)

    def extract_countries(self, wvs_selections):
        countries = []
        for response in wvs_selections:
            each_response = ast.literal_eval(response.strip("defaultdict(<class 'list'>, ").strip(")"))
            countries.extend(each_response.keys())
        return list(set(countries))

    def filter_common_countries(self, dollarstreet_countries = None, cvqa_countries = None):
        # Get the common countries between WVS and DollarStreet data/cvqa data which is not none
        if dollarstreet_countries is not None:
            common_countries = list(set(dollarstreet_countries).intersection(self.wvs_countries))
        elif cvqa_countries is not None:
            common_countries = list(set(cvqa_countries).intersection(self.wvs_countries))
        return common_countries

    # Note we only cosnider the family data for images
    def prepare_ds_family_data(self, dollarstreet_data, common_countries):
        return dollarstreet_data[
            dollarstreet_data['topics'].isin(['Family', 'Family snapshots']) &
            dollarstreet_data['country.name'].isin(common_countries) &
            dollarstreet_data['type'].isin(['image'])
        ]
    
    def prepare_cvqs_img_data(self, cvqa_data, common_countries):
        return cvqa_data[
            cvqa_data['country'].isin(common_countries)
        ]

